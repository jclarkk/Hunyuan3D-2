from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


class MoondreamPBRAssessmentPipeline:

    @classmethod
    def from_pretrained(cls):
        model_id = "vikhyatk/moondream2"
        revision = "2025-01-09"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return cls(model, tokenizer)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, image: Image.Image):
        enc_image = self.model.encode_image(image)

        prompt = "What should be the roughness factor for the texture of the following object? Answer with a float value between 0 to 1."

        roughness_answer = self.model.answer_question(enc_image, prompt, self.tokenizer)

        prompt = "What should be the metallic factor for the texture of the following object? Answer with a float value between 0 to 1."

        metallic_answer = self.model.answer_question(enc_image, prompt, self.tokenizer)

        return float(roughness_answer), float(metallic_answer)
