use neural_nets_toys::*;

fn main() {

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        let v = sigmoid(x);
        v * (1.0 - v)
    }

    let mut model = layers::attention_layer::Attention::<5,2>::random(
        -1.0..1.0,
        )
    .chain(
        layers::lnn_exp_layer::LNNLayer::<5,1,3>::random(
            sigmoid,
            -1.0..1.0,
            0.7
        ).with_derivative(sigmoid_derivative)
    )
    // .chain(layers::fc_layer::Layer::<3,1>::random(
    //     sigmoid,
    //     -1.0..1.0,
    // ).with_derivative(sigmoid_derivative))
    ;


    let data = [
        (
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0],
        ),
        (
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0],
        ),
        (
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0],
        ),
        (
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0],
        ),
        (
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0],
        ),
        (
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0],
        ),
    ];


    train(&mut model, &data, TrainParams {
        epochs: 1000,
        temperature: 0.1,
        cutoff: 0.1,
        fn_loss: |t,p| [
            t.iter().zip(p.iter())
            .map(|(t,p)| (t-p).powi(2))
            .sum::<f64>()
            ;1
        ],
    });

    for (x,_) in data.iter() {
        let y = model.forward(x, None::<&mut DefaultHelper>);
        println!("{:?} -> {:?}", x, y);
    }

    println!("Hello, world!");
}
