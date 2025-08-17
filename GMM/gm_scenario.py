# Stolen from USGS github
class gm_scenario:
        def __init__(self, Mw, dip, rake, width, 
                     R_jb, R_rup, R_x, 
                     vs30, vs30_flag,
                     z1p0, z2p5, z_hyp, z_tor,
                     SOF:int = 0, HW_flag: int = 0, R_y0:int = 0, region:int = 0):
                
                names = ['Mw', 'dip', 'rake', 'width', 
                         'R_jb', 'R_rup', 'R_x', 
                         'vs30', 'vs_flag', 
                         'z1p0', 'z2p5', ' z_hyp', 'z_tor',
                         'SOF', 'HW_flag', 'R_y0', 'region']
                

                self.Mw, self.dip, self.rake, self.width = Mw, dip, rake, width
                self.R_jb, self.R_rup, self.R_x = R_jb, R_rup, R_x
                self.vs30 = vs30
                self.vs30_flag = vs30_flag
                self.z1p0, self.z2p5, self.z_hyp, self.z_tor = z1p0, z2p5, z_hyp, z_tor
                self.SOF, self.HW_flag, self.R_y0, self.region = SOF, HW_flag, R_y0, region