import moondream as md
from PIL import Image


class MoondreamPipeline:

    @classmethod
    def from_pretrained(cls):
        pipe = md.vl(model="./weights/moondream-2b-int8.mf")
        return cls(pipe)

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, input_image: Image.Image) -> (float, float):
        # Input
        encoded_image = self.pipe.encode_image(input_image)

        # Prompts
        roughness_prompt = "Estimate the roughness intensity for this input image as used in a PBR texture map. " \
                            "Consider that highly reflective and smooth surfaces (like polished metal) should be closer to 0, " \
                            "while rough, matte surfaces should be closer to 1. Provide a single float value between 0 and 1."
        metalness_prompt = "Estimate the metalness intensity for this input image as used in a PBR texture map. " \
                            "Consider that metals should have a value close to 1, while non-metallic materials (such as plastic, wood, or fabric) should be near 0. " \
                            "Provide a single float value between 0 and 1."

        roughness_result = self.pipe.query(encoded_image, roughness_prompt)["answer"]
        metalness_result = self.pipe.query(encoded_image, metalness_prompt)["answer"]

        try:
            return float(roughness_result), float(metalness_result)
        except ValueError:
            print("Error: Could not convert roughness and metalness results to float, values are: ", roughness_result, metalness_result)
            return 0.5, 0.5
