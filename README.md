# Estimator Wrapper
This package is used to broadcast TensorFlow's estimators using spark


## Install
`pip install estimator-wrapper`


## Usage
### TF 1.X

```Python
from estimator_wrapper import EstimatorWrapper

estimator_wrapper = EstimatorWrapper(model_dir)

estimator_wrapper_broadcast = spark.sparkContext.broadcast(
    estimator_wrapper
)

@pandas_udf(T.StringType())
def inference(tfr):
    outputs = estimator_wrapper_broadcast.value.estimator(
        {"examples": tfr}
    )
    outputs = np.concatenate(
        (
            outputs["take_up"],
            outputs["redemption"],
            outputs["avg_basket_size"],
        ),
        axis=1,
    ).tolist()
    return pd.Series([json.dumps(out) for out in outputs])
```

### TF 2.X

```Python
from estimator_wrapper import EstimatorWrapper

estimator_wrapper = EstimatorWrapper(model_dir)

estimator_wrapper_broadcast = spark.sparkContext.broadcast(
    estimator_wrapper
)

@F.pandas_udf(T.StringType())
def inference(tfr):
    outputs = estimator_wrapper_bc.value.estimator.signatures[
        "serving_default"
    ](examples=tf.constant(tfr))
    return pd.Series(
        [
            json.dumps(out)
            for out in outputs["user_profile"].numpy().tolist()
        ]
    )
```