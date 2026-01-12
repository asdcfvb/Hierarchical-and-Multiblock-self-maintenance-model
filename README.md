> # Adaptive assessment of operating performance for multiprocess industrial process based on hierarchical and multiblock self-maintenance model
>
> This is a repository for studying the effectiveness of self maintaining deep learning methods in the domain of **performance assessment **. 
>
> This subject contains papers, codes, and data used for discussion in this article.
>
> ## 📝Papers
>
> > We list the papers according to the basic framework of the method, which can be roughly divided into **Offline phase**, **Online phase** and **Saved model parameters**
>
> ### Survey
>
> ***
>
>  - Adaptive assessment of operating performance for multiprocess industrial process based on hierarchical and multiblock self-maintenance model (At present, the paper is still under review in the Applied Intelligence journal and will be published as soon as it is published)
>
>  - 🔧Code
>
>
> > Our code is released at [[**Code link**]](https://github.com/asdcfvb/Hierarchical-and-Multiblock-self-maintenance-model/tree/main)
>
> You can check the structure framework by looking at `model.py` .
>
> If you want to see the details and settings of the code running, you can try to reproduce it by running `HMASCRAE_phase.py` in the `experiment` file.
>
> At the same time, we will place the offline trained model parameters `scaler_params_final. py` and `semi_Supervised_HMASCRAE.final. py`  in the `save_model.py` file for you to quickly reproduce the results in the paper in `HMASCRAE.online. py` without the need for retraining.
> 
><img width="529" height="577" alt="HMASCRAE" src="https://github.com/user-attachments/assets/5301c609-1784-42d7-a95a-f78177b382e5" />

> 
>
> ## 👜Data
>
> > The dataset of the hydrocarbon-liquid preparation process provided in this article is in the datasets file.
```
.
└── datasets
        ├── Alkylation.xlsx
        │── Alkylation - train.xlsx
        │── Alkylation - test.xlsx
        ├── Transalkylation.xlsx
        │── Transalkylation - train.xlsx
        ├── Transalkylation - test.xlsx
```
## :telephone:Contact
If you have any problem, please feel free to contact me.
Name: Honghe Du
Email address: [duhhlut@163.com](mailto:duhhlut@163.com)
