
## Performance Report

There are a total of 4 aspects in Sentihood dataset; general, price, safety and transit location. The tables below show the performance of the Roberta QA M model (with context from last 4 layers) for each aspect. We report the macro averaged precision, recall, f1-score and accuracy on the test set . 

The performance for safety aspect is the best across all the four metrics metrics, while general and transit location both have similar performance which is worse than safety and price. 
The low performance on general aspect can be attributed to the ambigious nature of the aspect definition itself, which makes it particularly hard to model. The transit location aspect is not expressed in a straightforward manner in the dataset, unlike price and safety aspect where keywords like "affordable", "expensive" and "safety"  are predominantly used. 

For price, safety and transit location the model is easily able predict "none" class, however for general aspect it is not able demonstrate the same performance. This again can be attributed to the ambiguous nature of this aspect. 

Across all the four aspects the performance for positive sentiment is much more than negative sentiment, across precision, f1-score and accuracy metrics. The difference between the recall for negative and positive sentiment is very small for safety aspect, and for transit location and price the recall for negative sentiment is higher. 

Although we observe (from the confusion matrix below) that misclassifications between positive and negative sentiment exist, they are comparitively less than the misclassifications between none and positive, negative. A point of failure for the model is that it fails to detect aspects in samples. This happens for 4% of the samples for general aspect and 2% of the samples for transit location aspect. This is again due to the fact that these aspects are not expressed in a straighforward manner in the dataset. 


### Aspect wise performance

| **aspect**       | **precision** | **recall** | **f1-score** | **acccuracy** |
|------------------|---------------|------------|--------------|---------------|
| general          |     0.7766    |   0.8276   |    0.7992    |     0.8648    |
| price            |     0.8066    |   0.9163   |    0.8484    |     0.9415    |
| safety           |     0.8251    |   0.9115   |    0.8642    |     0.9670    |
| transit location |     0.7553    |   0.8391   |    0.7910    |     0.9377    |

### General

| **sentiment** | **precision** | **recall** | **f1-score** | **support** |
|---------------|---------------|------------|--------------|-------------|
| negative      |     0.6250    |   0.7554   |     0.684    |     139     |
| none          |     0.9369    |   0.8840   |    0.9097    |     1293    |
| positive      |     0.7678    |   0.8434   |    0.8038    |     447     |

### Price

| **sentiment** | **precision** | **recall** | **f1-score** | **support** |
|-----------|---------------|------------|--------------|-------------|
| negative  |     0.6278    |   0.9790   |    0.7650    |     139     |
| none      |     0.9974    |   0.9459   |    0.9710    |     1628    |
| positive  |     0.7946    |   0.8241   |    0.8091    |     108     |

### Safety

| **sentiment** | **precision** | **recall** | **f1-score** | **support** |
|---------------|---------------|------------|--------------|-------------|
| negative      |     0.7157    |   0.8795   |    0.7892    |      83     |
| none          |     0.9923    |   0.9750   |    0.9836    |     1721    |
| positive      |     0.7674    |   0.8800   |    0.8199    |      75     |

### Transit location

| **sentiment** | **precision** | **recall** | **f1-score** | **support** |
|---------------|---------------|------------|--------------|-------------|
| negative      |     0.5614    |   0.7805   |    0.6531    |      41     |
| none          |     0.9755    |   0.9590   |    0.9672    |     1658    |
| positive      |     0.7292    |   0.7778   |    0.7527    |     180     |


### Confusion Matrix

|          | None | Positive | Negative |
|----------|------|----------|----------|
| None     | 5951 |    199   |    150   |
| Positive |  88  |    672   |    50    |
| Negative |  46  |    10    |    350   |
