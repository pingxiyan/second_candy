

class bbox(object):
    def __init__(self, other=None):
        if other is None:
            self.x0 = -1
            self.x1 = -1
            self.y0 = -1
            self.y1 = -1
            self.name = ""
        else:
            self.x0 = other.x0
            self.x1 = other.x1
            self.y0 = other.y0
            self.y1 = other.y1
            self.name = other.name

    @classmethod
    def fromvoc(cls, obj):
        ret = cls()
        ret.x0 = obj.bndbox.xmin
        ret.x1 = obj.bndbox.xmax
        ret.y0 = obj.bndbox.ymin
        ret.y1 = obj.bndbox.ymax
        ret.name = obj.name
        return ret

    def __contains__(self, m):
        return m[0] >= self.x0 and m[0] <= self.x1 and m[1] >= self.y0 and m[1] <= self.y1

    def corner(self, i):
        if i == 1: return (self.x0, self.y0)
        if i == 2: return (self.x1, self.y0)
        if i == 3: return (self.x1, self.y1)
        if i == 4: return (self.x0, self.y1)
        return (self.x0, self.y0)

    @property
    def tl(self):
        return (self.x0, self.y0)

    @tl.setter
    def tl(self, value):
        self.x0, self.y0 = value

    @property
    def br(self):
        return (self.x1, self.y1)

    @br.setter
    def br(self, value):
        self.x1, self.y1 = value

    @property
    def loc(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @loc.setter
    def loc(self, value):
        self.x0, self.y0, self.x1, self.y1 = value
        self.check_order()

    def check_order(self):
        if self.x0 > self.x1: self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1: self.y0, self.y1 = self.y1, self.y0

    def area(self):
        return max(0, (self.y1 - self.y0) * (self.x1 - self.x0))

    def __and__(self, other):
        xA = max(self.x0, other.x0)
        yA = max(self.y0, other.y0)
        xB = min(self.x1, other.x1)
        yB = min(self.y1, other.y1)
        return max(0, xB - xA + 1) * max(0, yB - yA + 1)

    def __or__(self, other):
        xA = max(self.x0, other.x0)
        yA = max(self.y0, other.y0)
        xB = min(self.x1, other.x1)
        yB = min(self.y1, other.y1)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return float(self.area() + other.area() - interArea)

    def IOU(self, other):
        return (self & other) / (self | other)

    def make_square(self):
        ret = bbox(self)
        w = ret.x1 - ret.x0
        h = ret.y1 - ret.y0

        d = abs(w - h)
        d0 = int(d / 2)
        d1 = d - d0

        if w > h:
            ret.y0 -= d0
            ret.y1 += d0
        else:
            ret.x0 -= d0
            ret.x1 += d1
        return ret
