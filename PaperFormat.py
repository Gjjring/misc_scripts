"""
Module implements a class for creating different paper sizes with different
units
"""

class PaperFormat:

    def __init__(self,**kwargs):
        self.Format = 'A4'
        self.Units = 'mm'
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self,key,value)

    def convertunits(self,size,unit1,unit2):
        assert unit1 == 'mm'

        if unit2 == 'cm':
            factor = 1e-1

        elif unit2 == 'inches':
            factor = 0.0393701

        return (size[0]*factor,size[1]*factor)

    def size(self):
        assert self.Format == 'A4'
        size = (210,297)

        if self.Units != 'mm':
            size = self.convertunits(size,'mm',self.Units)

        return size



if __name__ == "__main__":
    P = PaperFormat()
    print (P.size())
    P2 = PaperFormat(Units="inches")
    print (P2.size())
