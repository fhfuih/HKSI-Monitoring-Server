import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from models.utils import GPU, empty_cache

from . import conversation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model():
    path = "OpenGVLab/InternVL3-8B"
    model = (
        AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(GPU)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=True
    )
    return model, tokenizer


def get_generation_config():
    return dict(
        max_new_tokens=128,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def process_sample(images, input_size=448, max_num=1):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    for img in images:
        img = Image.fromarray(img)
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values.to(GPU, dtype=torch.bfloat16), num_patches_list


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def chat(
    model,
    tokenizer,
    pixel_values,
    question,
    generation_config,
    history=None,
    return_history=False,
    num_patches_list=None,
    IMG_START_TOKEN="<img>",
    IMG_END_TOKEN="</img>",
    IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    verbose=False,
):
    empty_cache()
    if history is None and pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    # template = get_conv_template('drowsiness_detection')
    # template.system_message = 'You are a drowsiness detection expert with general knowledge of the field and access to a vast database of research on drowsiness detection.'
    template = conversation.get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

    history = [] if history is None else history
    for old_question, old_answer in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f"dynamic ViT batch size: {image_bs}")

    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(GPU)
    attention_mask = model_inputs["attention_mask"].to(GPU)
    generation_config["eos_token_id"] = eos_token_id
    generation_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config,
    )
    response = tokenizer.batch_decode(
        generation_output.sequences, skip_special_tokens=True
    )[0]
    response = response.split(template.sep)[0].strip()

    # Process scores and tokens
    scores = generation_output.scores
    token_ids = generation_output.sequences[0]  # All output tokens
    processed_scores = process_scores_and_tokens(scores, token_ids, tokenizer)

    # dummy scores; should be the same as number of responses
    # processed_scores = [(1.0, '10') for _ in range(len(response))]

    history.append((question, response))
    if return_history:
        return response, processed_scores, history
    else:
        query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
        query_to_print = query_to_print.replace(
            f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
        )
        if verbose:
            print(query_to_print, response)
        return response, processed_scores


def process_scores_and_tokens(scores, token_ids, tokenizer, top_k=5):
    processed_generations = []
    score = scores[0]  # Use only the first token's scores
    token_id = token_ids[0]  # Use only the first token's scores
    # for score, token_id in zip(scores, token_ids):
    probs = F.softmax(4 * score[0], dim=-1)
    top_probs, top_indices = probs.topk(top_k)
    pairs = [
        (prob.item(), tokenizer.decode([idx.item()]))
        for prob, idx in zip(top_probs, top_indices)
    ]
    pairs.sort(reverse=True)

    # processed_generations.append(pairs)

    return pairs


def get_highest_prob(processed_scores):
    probs_array = torch.tensor([score[0] for score in processed_scores])
    normalized_probs = F.softmax(probs_array, dim=0)
    return torch.max(normalized_probs).item()
