
class DHM:
    def __init__(self, objective="Zeiss 63x", n_resin=1.5, prop_dist=None):
        if objective == "Zeiss 20x":
            self.pixel_pitch = [0.276, 0.276]  # 20x objective
            self.prop_dist = -20.0  # Distance of sensor to back focal plane of tube lens
            self.r0 = 304  # ORIGINAL: dhm.objective["dcRadius"]  # dc_radius: 304
        elif objective == "Zeiss 63x":
            self.pixel_pitch = [0.0869, 0.0869]  # 63x objective
            self.prop_dist = -20.0  # ToDo: Ändern - welche Einheit braucht man !
            self.r0 = 304  # noch nicht implementiert - weiß nicht wo reinhard das andere her hat # ToDo: Fragen wo das herkommt
        else:
            raise NotImplementedError(f"Objective {objective} not implemented!")
        self.wavelength = 0.000675  # wavelength of the laser
        self.n_resin = 1.5
        self.prop_dist = prop_dist

    def get_parameter(self):
        para_dict = {}
        return para_dict