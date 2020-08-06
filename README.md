# Doctor-Strange

### Here is a small preview
![dr-strange](https://user-images.githubusercontent.com/68480967/89542340-2edb5a00-d81d-11ea-8f8c-b90a621d754f.gif)

#### Full video: https://www.linkedin.com/posts/karan-owalekar_computervision-machinelearning-doctorstrange-activity-6664606986279223296-HNoz

> In this program, we use the haar cascade files to detect the presence of palm and fist in the frame.

> If it is detected, depending on what gesture me made an glowing aura disk image is added on the pfame on top of that palm/ fist.

> Using the addWeighted function of openCV we simply merge those images with frame.

> Black pixels have RGB values as (0, 0, 0) Hence all the black pixels are ignored and cant be seen on the frame leaving with the aura disk image.

> Using 2 images of green aura which are same but 45 degrees apart from each other, we display each one of them alternatively giving us an illusion of rotation.

> Using dlib we find the 6 pionts accors both eyes, connecting those points and giving them color also make it look visually cool.
