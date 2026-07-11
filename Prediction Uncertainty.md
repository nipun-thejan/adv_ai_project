## Useful resources on Uncertainty.

* [Quantifying the Uncertainty in Model Predictions](https://www.youtube.com/watch?v=-K8vDIyT3xY)  
* [https://www.youtube.com/watch?v=toTcf7tZK8c](https://www.youtube.com/watch?v=toTcf7tZK8c)  
* [https://www.youtube.com/watch?v=veYq6EWZyVc\&t=2597s](https://www.youtube.com/watch?v=veYq6EWZyVc&t=2597s)

* Important concept: Calibration error and the expected calibration error @12mins onwards.  
* Simple approaches to try @28 to 31 mins   
  * [Montecarlo dropout](https://proceedings.mlr.press/v48/gal16.html?trk=public_post_comment-text) (paper available for download at the bottom under related material)  
  * [Deep ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)  
  * [Hyperparameter ensembles](https://proceedings.neurips.cc/paper/2020/file/481fbfa59da2581098e841b7afc122f1-Paper.pdf)  
* [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://www.sciencedirect.com/science/article/pii/S1566253521001081?via%3Dihub)  
* [Evidential Deep Learning to Quantify Classification Uncertainty](https://proceedings.neurips.cc/paper_files/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf) (NIPS, 2018\)  
* [Improving model calibration with accuracy versus uncertainty optimization (NIPS, 2020\)](https://proceedings.nips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf) \- Code available.  
* [Rethinking Data Distillation: Do Not Overlook Calibration](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Rethinking_Data_Distillation_Do_Not_Overlook_Calibration_ICCV_2023_paper.pdf) (ICCV, 2023\)  
* [Dual Focal Loss for Calibration (ICML, 2023\)](https://proceedings.mlr.press/v202/tao23a/tao23a.pdf) \- Talks about post-hoc calibration too. \- Code available.  
* ML with rejection: [https://link.springer.com/article/10.1007/s10994-024-06534-x](https://link.springer.com/article/10.1007/s10994-024-06534-x) ([https://arxiv.org/pdf/2107.11277](https://arxiv.org/pdf/2107.11277))  
* [A survey of uncertainty in deep neural networks](https://link.springer.com/content/pdf/10.1007/s10462-023-10562-9.pdf) (2023, 1100+ citations) \- See section 4.1.2 Measuring model uncertainty in classification tasks \- Check OOD as well  
* [A Survey on Uncertainty Quantification Methods for Deep Learning](https://arxiv.org/pdf/2302.13425) (2023) \- See section 5.3.4 for Evaluation Metrics for calibration.  
* [Brier score](https://library.virginia.edu/data/articles/a-brief-on-brier-scores)

## ML with rejection

* ML with rejection: [https://link.springer.com/article/10.1007/s10994-024-06534-x](https://link.springer.com/article/10.1007/s10994-024-06534-x) ([https://arxiv.org/pdf/2107.11277](https://arxiv.org/pdf/2107.11277))  
  * P16: Confidence as 1-var\[h1(x), h2(x),..hm(x)\] from multiple prediction models. This can be thresholded to decide what to reject. Variance needs to be normalized to \[0,1\].  
  * P16: Ensemble agreement is also an option  
  * P21: Integrated rejector options  
* [Reliable Multilabel Classification: Prediction with Partial Abstention](https://ojs.aaai.org/index.php/AAAI/article/view/5972/5828#:~:text=For%20such%20problems%2C%20the%20idea,which%20it%20is%20certain%20enough.) (2020 AAAI Conference on Artificial Intelligence 200+ h-index)  
* [Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf) (MLR, 2020\)  
* [Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.pdf) (CVPR, 2020\)

**Post-hoc calibration.**

* [Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence (NIPS, 2021\)](https://proceedings.neurips.cc/paper/2021/file/61f3a6dbc9120ea78ef75544826c814e-Paper.pdf)  
* [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) (ICML, 2017, 5000+ citations)


  