#!/bin/bash
curl -o learning_protein_motif_Hsp70_data.tar.gz http://www.phys.ens.fr/~monasson/learning_protein_motif_Hsp70_data.tar.gz
tar -xvf learning_protein_motif_Hsp70_data.tar.gz
mkdir data/Hsp70/
mv learning_protein_motif_Hsp70_data/Hsp70_info.data data/Hsp70/
mv learning_protein_motif_Hsp70_data/Hsp70_protein_MSA.fasta data/Hsp70/
mv learning_protein_motif_Hsp70_data/RBM_Hsp70_Protein.data models/
rm -r learning_protein_motif_Hsp70_data
rm learning_protein_motif_Hsp70_data.tar.gz