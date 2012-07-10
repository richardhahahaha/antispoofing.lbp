#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Wed 20 Jul 11:02:50 2011 

"""Classes that help mapping eye-coordinates from face bounding boxes
   see "Anthropometry of the Head and Face" L.G. Farkas, Raven Press
"""

class Anthropometry19x19:
  """A helper to convert KeyLemon (MCT face localization) bounding boxes to
  eye coordinates"""

  # Some constants we need for the job
  PUPIL_SE = (33.4+31.4)/2 #mm pupil-facial middle distance, p.275
  PUPIL_SE_SD = 2.0 #larger for males
  PUPIL_OS = (23.3+23.0)/2 #mm pupil-eyebrow distance, p.280
  PUPIL_OS_SD = 3.3 #mm larger for males
  N_STO = (76.6+69.4)/2.0 #mm from eye top to mouth, p.255
  N_STO_SD = 4.0 #larger for males
  CH_CH = (54.5+50.2)/2.0 #mm mouth width, p.303 
  CH_CH_SD = 3.5 #larger for females
  EX_EX = (91.2+87.8)/2.0 #mm outside eye corners (left and right), p.272
  EX_EX_SD = 3.2 #larger for females
  EN_EN = (33.3+31.8)/2.0 #mm inside eye corners (left and right), p.272
  EN_EN_SD = 2.7 #larger for males

  # Parameters for the KeyLemon model, taken from Torch3vision2.1 following
  # Sébastien's advice (c.f. mail on eye and mouth positions given bounding box
  # exchanged around the 21st./february/2011)
  MODEL_WIDTH = 19. #pixels, normalized model width
  D_EYES = 10. #pixels, normalized pixel distance between pupils
  Y_UPPER = 5. #pixels, normalized distance between head-top and eye-pupils

  def __init__(self, bbox):
    """Starts a new object with a bounding box"""
    self.bb = bbox
    if bbox is not None and self.bb.is_valid():
      self.ratio = self.bb.width / self.MODEL_WIDTH
      self.anthropo_ratio = (self.D_EYES*self.ratio)/(2*self.PUPIL_SE)
    else:
      from . import BoundingBox
      self.bb = BoundingBox(0,0,0,0)

  def eye_centers(self):
    """Returns the eye centers coordinates"""

    if not self.bb.is_valid(): return ((None, None), (None, None))

    Rx = (self.ratio * (self.D_EYES + self.MODEL_WIDTH) / 2) + self.bb.x
    Lx = Rx - (self.D_EYES * self.ratio)
    y = self.bb.y + (self.ratio * self.Y_UPPER)
    return ((round(Lx), round(y)), (round(Rx), round(y)))

  def face_center(self):
    """Returns the mid distance between eye brows and mouth top"""

    if not self.bb.is_valid(): return (None, None)

    x = (self.ratio * (self.D_EYES + self.MODEL_WIDTH) / 2.) + self.bb.x
    x -= (self.D_EYES * self.ratio) / 2.
    y = self.bb.y + (self.ratio * self.Y_UPPER)
    y += (self.ratio * self.D_EYES) / 4.
    return (round(x), round(y))

  def ear_centers(self):
    """Returns the ear centers left, right"""

    if not self.bb.is_valid(): return ((None, None), (None, None))

    Rx = self.bb.x + self.bb.width #+ (self.ratio * self.D_EYES) / 5.0
    Lx = self.bb.x #- (self.ratio * self.D_EYES) / 5.0
    y = self.bb.y + (self.ratio * self.Y_UPPER)
    y += (self.ratio * self.D_EYES) / 2.
    return ((round(Lx), round(y)), (round(Rx), round(y)))

  def mouth_bbox(self):
    """Returns the mouth bounding box (UNTESTED!) """

    from . import BoundingBox

    if not self.bb.is_valid(): BoundingBox(0, 0, 0, 0)

    Mx = self.bb.x + (self.bb.width/2.)
    Eye_y = self.bb.y + (self.ratio * self.Y_UPPER)
    My = Eye_y + ((self.N_STO-(self.PUPIL_OS/2.)) * self.anthropo_ratio)
    Mwidth = (self.CH_CH * self.anthropo_ratio)
    Mheight = 30. * self.anthropo_ratio #guessed
    return BoundingBox(round(Mx), round(My), round(Mwidth), round(Mheight))

  def eye_area(self):
    """Returns a bounding box to the eyes area"""

    from . import BoundingBox
    
    if not self.bb.is_valid(): BoundingBox(0, 0, 0, 0)

    Eye_y = self.bb.y + (self.ratio * self.Y_UPPER)
    Eye_x = self.bb.x + (self.bb.width / 2.) # eyes center
    Box_width = (self.EX_EX + 8.*self.EX_EX_SD) * self.anthropo_ratio
    Box_height = (self.PUPIL_OS + self.PUPIL_OS_SD) * 1.2 * self.anthropo_ratio
    Box_x = Eye_x - (Box_width/2.)
    Box_y = Eye_y - (Box_height/2.)
    return BoundingBox(round(Box_x), round(Box_y), round(Box_width), round(Box_height))
