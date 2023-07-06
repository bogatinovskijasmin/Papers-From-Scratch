### Recommendation Methods

The recommendation methods are one of the most frequently utilized methods in the industry. 
They are commonly part of recommendation systems to recommend different items to users.

The enabler of recommendation methods is the existing interaction between the users and items, i.e., 
some user, at one point in time, obtained some item. To enable recommendations in practice, it is 
further necessary that there exist a set of users that interacted at least once with at least one item. 
That way, by determining similarities between users and the types of their interactions with the items, one can provide 
useful recommendations. 

Generally speaking, there are two types of feedback: 
1. **Explicit**; Where the users rate the item (e.g., thumbs up, star rating etc.).
2. **Implicit**; Far more common. In this case, users are reluctant to explicitly state their opinion, and other queues (e.g., origin, day of the year, holiday, clicks etc) are used to infer the opinions of the users.

There are two properties of the problem: 
1. **Sparsity**; Commonly, there are millions of users and items, and very few users interact with very few items; 
2. **Cold-start**; There are many new items and many new users appearing continuously. The recommendation problem usually faces the challenge of "cold-start" as it is frequent to have new items, as well as new users.  

Most of the work is based on the matrix factorization on the score-alike matrix (S = HQ). Commonly the Q matrix contains the 
not necessarily interpretable "properties" of the items, while the H matrix refers to the preference of the user for certain items.

It is important to note that the idea of **negative sampling** is commonly used for training these models, especially for 
the **implicit** group of methods. 

Common losses used for learning are: 
1. Margin loss;
2. Bayesian Personalized Ranking Loss; 

This project implements the following approaches:
1. Matrix Factorization; **Matrix factorization techniques for recommender systems.** Koren, Y., Bell, R., & Volinsky, C. (2009). Computer, pp. 30–37.
2. AutoRec: **Autorec: autoencoders meet collaborative filtering.**  Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015). Proceedings of the 24th International Conference on World Wide Web (pp. 111–112).
3. Factorization Machines; **Factorization machines.** Rendle, S. (2010). 2010 IEEE International Conference on Data Mining (pp. 995–1000).
4. Deep Factorization Machines (DeepFM); **Deepfm: a factorization-machine based neural network for ctr prediction.**  Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). Proceedings of the 26th International Joint Conference on Artificial Intelligence (pp. 1725–1731).

As a dataset, MovieLens (with given ranking scores) is used for the methods with explicit learning. The dataset "Click Through Rate" is used for learning the models for implicit recommendations.
