import math
import mmap
import os

class Deserializer:
    def __init__(self, file):
        self.file = file
        self.mm = mmap.mmap(file.fileno(), 0)
        self.index = 0
        self.file_len = os.fstat(file.fileno()).st_size

    def close(self):
        self.mm.close()

    def skip(self, count):
        ret = self.index
        self.index += count
        return ret

    def read_tensor(self, mark, l):
        return self.mm[mark:mark+l]

    def read_int(self):
        ch = self.mm[self.index]
        is_negative = ch & 0x80
        size = ch & 0x0F
        val = 0
        self.index += 1

        for i in range(self.index + size - 1, self.index - 1, -1):
            val <<= 8
            val |= self.mm[i]

        self.index += size
        if is_negative:
            val *= -1

        return val

    def read_int4(self):
        i1 = self.read_int()
        i2 = self.read_int()
        i3 = self.read_int()
        i4 = self.read_int()
        return i1, i2, i3, i4

    def read_int6(self):
        i1 = self.read_int()
        i2 = self.read_int()
        i3 = self.read_int()
        i4 = self.read_int()
        i5 = self.read_int()
        i6 = self.read_int()
        return i1, i2, i3, i4, i5, i6

    def read_uint(self):
        ch = self.mm[self.index]
        size = ch & 0x8F
        val = 0
        self.index += 1

        for i in range(self.index + size - 1, self.index - 1, -1):
            val <<= 8
            val |= self.mm[i]

        self.index += size

        return val

    def read_float(self):
        mantissa = self.read_int()
        exponent = self.read_int()
        return math.ldexp(mantissa, exponent)

    def read_float4(self):
        f1 = self.read_float()
        f2 = self.read_float()
        f3 = self.read_float()
        f4 = self.read_float()
        return f1, f2, f3, f4

    def readString(self, l):
        val = self.mm[self.index:(self.index + l)].decode()
        self.index += l
        return val
