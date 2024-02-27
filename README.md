# ADVEI-Paper-2024.3-Degradation-path-approximation-for-remaining-useful-life-prediction

For remaining useful life prediction tasks, we propose a highly easy-to-understand and brand-new solution way for the data-driven RUL prediction, Degradation Path Approximation (DPA). Many research directions on DPA can be further studied. DPA is expert knowledge-free and can perform excellently in less degradation data, even a single historical degradation path.  We theoretically analyzed and proved the feasibility of DPA. A series of comparative and ablation experiments on Li-ion prismatic cells, milling cutters, 18650 Li-ion batteries, and IGBT are conducted to demonstrate the superiority of DPA and the importance of DPA variables, respectively. 

Paper: Degradation path approximation for remaining useful life estimation

The website of the paper：https://www.sciencedirect.com/science/article/pii/S1474034624000703?via%3Dihub




# Easy to reproduce all details of our paper successfully
The downloaded compressed package can reproduce our method. To make code easy to run successfully, we debug the files carefully. Generally speaking, if environments are satisfied, you can directly run all the xxx.py files inside after decompressing the compressed package without changing any code.
Note: The complete IGBT data set has been released.

(1) Download and rename xxx.zip to DPA.zip (Rename to avoid errors caused by long directories)

(2) Unzip DPA.zip

(3) Run any xxx.py directly


# Paper of Code and Citation
(1) To better understand our code, please read our paper.

Paper: Degradation path approximation for remaining useful life estimation

The website of the paper：https://www.sciencedirect.com/science/article/pii/S1474034624000703?via%3Dihub

(2) Please cite this paper and the original source of the dataset when using the code for academic purposes.

GB/T 7714: 
Fan L, Lin W, Chen X, et al. Degradation path approximation for remaining useful life estimation[J]. Advanced Engineering Informatics, 2024, 60: 102422.

BibTex:
@article{fan2024degradation,
  title={Degradation path approximation for remaining useful life estimation},
  author={Fan, Linchuan and Lin, Wenyi and Chen, Xiaolong and Yin, Hongpeng and Chai, Yi},
  journal={Advanced Engineering Informatics},
  volume={60},
  pages={102422},
  year={2024},
  publisher={Elsevier}
}


# Relationship between Code and Paper

 (1) Section 2.2  Degradation path approximation
 
 :code\comparison experiments\xxx\DPA_our_method\DPA

 (2) TABLE 4 TaFCN
 
 :code\comparison experiments\xxx\tafcn2022\TaFCN


# Environment and Acknowledgement:

(1) Environment:

    
scipy                     1.5.2
    
pandas                    1.0.5
    
numpy                     1.19.1


(2) Acknowledgement: 
Thanks for the following references sincerely.
   
[1] M. Pecht, Calce battery group, 2017. 
[2]Prognostics and Health Management Society, Prognostics and health management society, PHM data challenge 2010, 2010.
[3] B. Saha, K. Goebel, Battery Data Set, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, 2007. 
[4] Greg Sonnenfeld, Kai Goebel, Jose R. Celaya, An agile accelerated aging, characterization and scenario simulation system for gate controlled power transistors, in: 2008 IEEE AUTOTESTCON, IEEE, 2008, pp. 208–215. 
  
github：https://github.com/foryichuanqi/RESS-Paper-2022.09-Remaining-useful-life-prediction-by-TaFCN
