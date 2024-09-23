
class DHM:
    def __init__(self, objective="Zeiss 63x", n_resin=1.5, prop_dist=None):
        if objective == "Zeiss 20x":
            self.pixel_pitch = [0.276e-6, 0.276e-6]  # 20x objective [m]
            self.prop_dist = 0.02  # Distance of sensor to back focal plane of tube lens [m]
            self.r0 = 304  # ORIGINAL: dhm.objective["dcRadius"]  # dc_radius: 304
        elif objective == "Zeiss 63x":
            self.pixel_pitch = [0.0869e-6, 0.0869e-6]  # 63x objective [m]
            self.prop_dist = 0.002  # [m]
            self.r0 = 304  # noch nicht implementiert - weiß nicht wo reinhard das andere her hat # ToDo: Fragen wo das herkommt
        else:
            raise NotImplementedError(f"Objective {objective} not implemented!")
        self.wavelength = 675E-9  # wavelength of the laser [m]
        self.n_resin = 1.5

    def get_parameter(self):
        para_dict = {}
        return para_dict