TARGETS = 1_Introduction/1_intro.pdf \
	2_Linear_Classifiers/2_linear_classifiers.pdf \
	4_SVMs/4_SVMs.pdf \
	6_Neural_Networks/6_NNs.pdf \
	8_Deep_Learning/8_Deep_Learning.pdf \
	projects/MiniNN.pdf \
	projects/readme.pdf \
	1_Introduction/HW1.pdf \
	3_Decision_Trees/hw3.pdf \
	3_Decision_Trees/3_decision_trees.pdf \
	4_SVMs/hw4.pdf \
	5_Regression/5_regression.pdf \
	7_Clustering/7_Clustering.pdf \
	jupyterhub/howto_students.pdf \
	projects/organic_chemistry.pdf \
	for_TAs.pdf \
	README.pdf \
	syllabus.pdf

6_FIGS_DIR = 6_Neural_Networks/figs

FIGURES = $(6_FIGS_DIR)/one_neuron.pdf \
	$(6_FIGS_DIR)/one_neuron_2.pdf \
	$(6_FIGS_DIR)/two_hidden_layers.pdf \
	$(6_FIGS_DIR)/example.pdf \
	$(6_FIGS_DIR)/layers.pdf \
	$(6_FIGS_DIR)/backprop_two_hidden_layer.pdf \
	7_Clustering/dbscan.pdf

.PHONY: all
all: $(FIGURES) $(TARGETS)

.NOTPARALLEL:

%.pdf: %.md
	cd $(shell dirname "$<") && pandoc -t beamer $(shell basename "$<") -o $(shell basename "$@")

$(6_FIGS_DIR)/%.pdf : $(6_FIGS_DIR)/%.tex
	pdflatex --output-directory $(6_FIGS_DIR) "$<"

3_Decision_Trees/3_decision_trees.pdf : 3_Decision_Trees/3_decision_trees.md
	cd 3_Decision_Trees && pandoc -t beamer $(shell basename "$<") -o $(shell basename "$@") --pdf-engine=xelatex

7_Clustering/dbscan.pdf : 7_CLustering/dbscan.tex
		pdflatex --output-directory 7_Clustering "$<"

.PHONY: clean
clean:
	rm -f $(TARGETS) $(FIGURES)