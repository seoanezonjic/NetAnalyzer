class Performancer:
	def __init__(self):
		self.control = {}

	def load_control(self, ref_array):
		for pair in ref_array:
			node1, node2 = pair
			if node2 != '-':
				query = self.control.get(node1)
				if query == None:
					self.control[node1] = [node2]
				else:
					query.append(node2)

	# Pandey 2007, Association Analysis-based Transformations for Protein Interaction Networks: A Function Prediction Case Study
	def get_pred_rec(self, predictions, cut_number = 100, top_number = 10000):
		preds, limits = self.load_prediction(predictions)
		cuts = self.get_cuts(limits, cut_number)
		performance = [] #cut, pred, rec
		for cut in cuts:
			prec, rec = self.pred_rec(preds, cut, top_number)
			performance.append([cut, prec, rec])
		return performance

	def load_prediction(self, pairs_array):
		pred = {}
		min = None
		max = None
		for rec in pairs_array:
			key, label, score = rec
			if min != None and max != None:
				if score < min: min = score 
				if score > max: max = score 
			else:
				min = score; max = score
			query = pred.get(key)
			if query == None:
				pred[key] = [[label], [score]]
			else:
				query[0].append(label)
				query[1].append(score)
		return pred, [min, max]

	def pred_rec(self, preds, cut, top):
		predicted_labels = 0 #m
		true_labels = 0 #n
		common_labels = 0 # k
		for ctrl in self.control:
			key, c_labels = ctrl
			true_labels += len(c_labels) #n
			pred_info = preds[key]
			if pred_info != None:
				labels, scores = pred_info
				reliable_labels = self.get_reliable_labels(labels, scores, cut, top)
				predicted_labels += len(reliable_labels) #m
				common_labels += len(set(c_labels) & set(reliable_labels)) #k
		if predicted_labels > 0:
			prec = common_labels/predicted_labels
		else:
			prec = 0.0

		if true_labels > 0:
			rec = common_labels/true_labels
		else:
			rec = 0.0

		return prec, rec


	def get_cuts(self, limits, n_cuts):
		min, max = limits
		span = (max - min)/n_cuts
		cut = min
		cuts = [ cut + n * span for n in range(n_cuts + 1) ]
		return cuts

	def get_reliable_labels(self, labels, scores, cut, top):
		reliable_labels = [ [labels[i], score] for i, score in enumerable(scores) if score >= cut ]
		def _(e):
			return e[1]
		reliable_labels.sort(reverse=True, key=_)
		reliable_labels = [ pred[0] for pred in reliable_labels[0:top-1:1] ]
		return reliable_labels
