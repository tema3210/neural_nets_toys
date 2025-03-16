use neural_nets_toys::*;

fn main() {

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        let v = sigmoid(x);
        v * (1.0 - v)
    }

    let mut model = attention::Attention::<5,2>::random(
        -1.0..1.0,
        )
    .chain(
        lnn_exp::LNNLayer::<5,3,3>::random(
            sigmoid,
            -1.0..1.0,
        ).with_derivative(sigmoid_derivative)
    )
    .chain(fc_layer::Layer::<3,1>::random(
        sigmoid,
        -1.0..1.0,
    ).with_derivative(sigmoid_derivative));


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
        let y = model.forward(x);
        println!("{:?} -> {:?}", x, y);
    }

    println!("Hello, world!");
}
