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
        # Inputs
        prompt = 'What should be the {texture_type} intensity for this input image in a PBR texture map? Answer with a float value between 0 and 1'
        encoded_image = self.pipe.encode_image(input_image)

        roughness_prompt = prompt.format(texture_type="roughness")
        metalness_prompt = prompt.format(texture_type="metalness")

        roughness_result = self.pipe.query(encoded_image, roughness_prompt)["answer"]
        metalness_result = self.pipe.query(encoded_image, metalness_prompt)["answer"]

        try:
            return float(roughness_result), float(metalness_result)
        except ValueError:
            print("Error: Could not convert roughness and metalness results to float, values are: ", roughness_result, metalness_result)
            return 0.5, 0.5
