class NotComputedProperlyException(Exception):
	def __init__(self,x):
		Exception.__init__(self, x+" not computed. Please run the attack() method again before calling this method.")


class NotRunProperlyException(Exception):
	def __init__(self,x):
		Exception.__init__(self, "report() or print_report() methods can be run only after a successfull run of "+x+"() method.")


class AdversariesNotComputedProperlyException(NotComputedProperlyException):
	def __init__(self):
		super().__init__("Adversaries")


class PerturbationNotComputedProperlyException(NotComputedProperlyException):
	def __init__(self):
		super().__init__("Perturbation")


class DetectNotRunProperlyException(NotRunProperlyException):
	def __init__(self):
		super().__init__("detect")

class MitigateNotRunProperlyException(NotRunProperlyException):
	def __init__(self):
		super().__init__("mitigate")


class TrainingDataNotProvidedException(Exception):
	def __init__(self):
		Exception.__init__(self, "This method requires some training data too. Please provide it while instantiating the class object.")

class TrueLabelsNotProvidedException(Exception):
	def __init__(self):
		Exception.__init__(self, "In order to run this method you need to provide true labels first.")

class DatasetNotLoadedException(Exception):
	def __init__(self):
		Exception.__init__(self, "get_info() method can be run only when a Dataset is successfully loaded.")