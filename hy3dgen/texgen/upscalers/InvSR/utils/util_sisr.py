import cv2

def modcrop(im, sf):
    h, w = im.shape[:2]
    h -= (h % sf)
    w -= (w % sf)
    return im[:h, :w,]

#-----------------------------------------Transform--------------------------------------------
class Bicubic:
    def __init__(self, scale=None, out_shape=None, matlab_mode=True):
        self.scale = scale
        self.out_shape = out_shape

    def __call__(self, im):
        out = cv2.resize(
                im,
                dsize=self.out_shape,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_CUBIC,
                )
        return out
