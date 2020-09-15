SHELL := /bin/bash
PACHCTL := pachctl
KUBECTL := kubectl

bc-single-pipeline:
	$(PACHCTL) create repo models
	$(PACHCTL) create repo sample_data
	$(PACHCTL) put file -r models@master:/ -f models/
	$(PACHCTL) put file -r sample_data@master:/ -f sample_data/
	$(PACHCTL) create pipeline -f pachyderm/single_stage/bc_classification.json 

bc-multi:
	$(PACHCTL) create repo models
	$(PACHCTL) create repo sample_data
	$(PACHCTL) put file -r models@master:/ -f models/
	$(PACHCTL) put file -r sample_data@master:/ -f sample_data/
	$(PACHCTL) create pipeline -f pachyderm/multi-stage/crop.json
	$(PACHCTL) create pipeline -f pachyderm/multi-stage/extract_centers.json
	$(PACHCTL) create pipeline -f pachyderm/multi-stage/generate_heatmaps.json
	$(PACHCTL) create pipeline -f pachyderm/multi-stage/classify.json

bc-update:
	$(PACHCTL) update pipeline -f pachyderm/multi-stage/crop.json
	$(PACHCTL) update pipeline -f pachyderm/multi-stage/extract_centers.json
	$(PACHCTL) update pipeline -f pachyderm/multi-stage/generate_heatmaps.json
	$(PACHCTL) update pipeline -f pachyderm/multi-stage/classify.json

bc-clean:
	$(PACHCTL) delete pipeline bc_classification
	$(PACHCTL) delete pipeline bc_classification_cpu
	$(PACHCTL) delete pipeline classify
	$(PACHCTL) delete pipeline generate_heatmaps
	$(PACHCTL) delete pipeline extract_centers
	$(PACHCTL) delete pipeline crop
	$(PACHCTL) delete repo sample_data
	$(PACHCTL) delete repo models