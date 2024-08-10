from PIL import Image


image=Image.open('kafu.jpeg')
image = image.convert('RGB')
image=image.resize((224,224))


print(image.size)