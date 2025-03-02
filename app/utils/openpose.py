from openpose import pyopenpose as op

class OPUtils:
    # Configura OpenPose
    def __init__(self, params):
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    # Deinizializza OpenPose
    def __del__(self):
        self.opWrapper.stop()
    
    # Processa i frame con OpenPose
    def process_frame(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        return datum