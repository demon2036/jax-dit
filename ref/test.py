from diffusers import DiTPipeline, DPMSolverSinglestepScheduler,DDIMScheduler
import torch
import matplotlib.pyplot as plt

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)

print(pipe.scheduler)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# pick words from Imagenet class labels
pipe.labels  # to print all available words
# print(pipe.labels)

# pick words that exist in ImageNet
words = ["white shark", "umbrella"]

class_ids = pipe.get_label_ids(words)

print(class_ids)




generator = torch.manual_seed(33)
output = pipe(class_labels=class_ids, num_inference_steps=100, generator=generator)

image = output.images[1]  # label 'white shark'


print(pipe.scheduler.config)

plt.imshow(image)
# print(image)
plt.show()