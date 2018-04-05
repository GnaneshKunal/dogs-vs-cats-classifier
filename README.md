# dogs-vs-cats-classifier
Dogs vs Cats classification using Transfer Learning and PyTorch

## DATASET

The dataset can be obtained from [here](https://www.kaggle.com/c/dogs-vs-cats/)

## Model
The Convolutional Neural Network will be trained on ResNet18. Transfer Learning is used. A pretrained ResNet18 model is initialied as a fixed feature extractor and training a softmax classifier at the end.

## Prediction

<div align="center">
  <img src="https://www.dropbox.com/s/k842le5vb74og5y/cnn1.png?raw=1"><br><br>
</div>


## Note
The dataset must have the following directory structure:

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

If you've trained and saved the model already, then comment the training phase to not to train the model again.
```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

## Useful Links
* [Transfer Learning](https://cs231n.github.io/transfer-learning/)
* [Resnet](https://arxiv.org/pdf/1512.03385.pdf)
* [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)
* [PyTorch](http://pytorch.org/docs/0.3.1/)

## Confusing the CNN 

<div align="center">
  <img src="https://www.dropbox.com/s/31p6pfvd58fld4f/cnn2.png?raw=1"><br><br>
</div>


## License

MIT License (MIT)

