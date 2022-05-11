import numpy
import imageio
import io

image = imageio.imread('D:/cats/3.jpg')
print(image.shape)

red_image = image
red_image[:, :, 1] = 0
red_image[:, :, 2] = 0
imageio.imwrite('D:/cats/rgb/red_image.jpg', red_image)


green_image =imageio.imread('D:/cats/3.jpg')
green_image[:, :, 0] = 0
green_image[:, :, 2] = 0
imageio.imwrite('D:/cats/rgb/green_image.jpg', green_image)


blue_image = imageio.imread('D:/cats/3.jpg')
blue_image[:, :, 0] = 0
blue_image[:, :, 1] = 0
imageio.imwrite('D:/cats/rgb/blue_image.jpg', blue_image)

red_image2 = imageio.imread('D:/cats/1.jpg')
red_image2[:, :, 1] = 0
red_image2[:, :, 2] = 0
imageio.imwrite('D:/cats/rgb/red_image2.jpg', red_image2)


green_image2 =imageio.imread('D:/cats/1.jpg')
green_image2[:, :, 0] = 0
green_image2[:, :, 2] = 0
imageio.imwrite('D:/cats/rgb/green_image2.jpg', green_image2)


blue_image2 = imageio.imread('D:/cats/1.jpg')
blue_image2[:, :, 0] = 0
blue_image2[:, :, 1] = 0
imageio.imwrite('D:/cats/rgb/blue_image2.jpg', blue_image2)
