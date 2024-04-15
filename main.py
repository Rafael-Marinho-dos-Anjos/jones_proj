from actions_classificator.classify_action import look_person_activitie
from PIL import Image


image_path = r"data\sample_images\will-smith-fam-oscars-red-carpet-2022-billboard-1548.jpg"
image = Image(image_path)
image = image.convert('RGB')

look_person_activitie(image, 50)
