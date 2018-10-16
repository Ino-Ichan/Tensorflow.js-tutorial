
// ここからc3jsの練習
var data1 = ['data1', 30, 200, 100, 400, 150, 250];
var chart = c3.generate({
    bindto: '#chart',
    data: {
        columns: [
            data1,
            ['data2', 50, 20, 10, 40, 15, 25]
        ]
    }
});

var chart2 = c3.generate({
    bindto: "#chart2",
    data: {
        columns: [
            ["data1", 30, 200, 100, 400, 150, 250],
            ["data2", 50, 20, 10, 40, 15, 25]
        ],
        axes: {
            data2: "y2"
        },
        types: {
            data2: "bar"
        }
    },
    axis: {
        y: {
            label: {
                text: "Y1 label",
                position: "outer-middle"
            }
        },
        y2: {
            show: true,
            label: {
                text: "これはY2ラベルです♬",
                position: "inner-middle"
            }
        }
    }
});

function loading(){
    chart2.load({
        columns: [
            ["data3", 65, 800, 13, 654, 30, 500, 420]
        ]
    });
}

function unload(){
    chart2.unload({
        ids: ["data3"]
    });
}

function show_d2() {
    chart2.show(["data2"]);
}

function hide_d2(){
    chart2.hide(["data2"]);
}



//ここからTFJSのサンプル

// Notice there is no 'import' statement. 'tf' is available on the index-page
// because of the script tag above.

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
// Use the model to do inference on a data point the model hasn't seen before:
// Open the browser devtools to see the output
model.predict(tf.tensor2d([5], [1, 1])).print();
});

var result = document.getElementById('result');
result.innerHTML = model.predict(tf.tensor2d([5], [1, 1]));


// ここからtfjsの練習

// サンプルデータの作成
// 正しい係数　a=-0.8, b=-0.2, c=0.9, d=0.5


function tensor1d_to_array (tensor1d){
    var array = [];
    for (var i = 0; i < tensor1d.size; i++){
        array.push(tensor1d.get(i));
    }
    return array;
}



function true_func(x){
    const a = -0.8;
    const b = -0.2;
    const c = 0.9;
    const d = 0.5
    return a * x ** 3 + b * x ** 2 + c * x + d;
};

// xの値 tensor
var x_val = tf.range(-1.0, 1.0, 0.01);

//yの値
var y_val_with_noise = [];

x_val_array = tensor1d_to_array(x_val);
x_val_array.forEach(function(x){
    var noise = tf.randomUniform([1], -0.1, 0.1).asScalar().get();
    y_val_with_noise.push(true_func(x) + noise);
});

//yの値 tensor
var y_val = tf.tensor1d(y_val_with_noise);


var debug = document.getElementById('debug');
debug.innerHTML = "y_val : " + y_val_with_noise.length;

// 実験で使う値はできた

//グラフにするために値の整形
var x_plot = ["t_data_x"];
x_val_array.forEach((i) => {
    x_plot.push(i);    
});

var y_plot = ["true_data"];
y_val_with_noise.forEach((i)=>{
    y_plot.push(i);
});

var debug2 = document.getElementById('debug2');
debug2.innerHTML = "y_plot length : " + y_plot.length + "  x_plot length : " + x_plot.length;

var true_data = c3.generate({
    bindto: "#true_data",
    size: {
        height: 500,
        width: 500
    },
    data: {
        xs: {
            true_data: "t_data_x"
        },
        columns:
            [
                x_plot,
                y_plot
            ],
        type: "scatter"
    },
    axis: {
        x: {
            label: 'x',
            tick: {
                fit: false
            }
        }
    }

});

//===========ここまででオリジナルデータの描画おっけ===========


//------------ここからTensorflow.jsのTutorial-----------
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

var a_result = a.get();
var b_result = b.get();
var c_result = c.get();
var d_result = d.get();

function predict(x) {
    return tf.tidy(()=>{
        return a.mul(x.pow(tf.scalar(3)))
            .add(b.mul(x.pow(tf.scalar(2))))
            .add(c.mul(x))
            .add(d);
    });
}

y_first_predict = ["y_predict_1"];
x_val_array.forEach((x) => {
    return tf.tidy(()=>{
        var x_tf = tf.scalar(x);
        y_first_predict.push(predict(x_tf).asScalar().get());
    });    
});

var tfjs_fig2 = c3.generate({
    bindto: "#tfjs_fig2",
    size: {
        height: 500,
        width: 500
    },
    data: {
        xs: {
            true_data: "t_data_x",
            y_predict_1: "t_data_x"
        },
        columns: [
            x_plot,
            y_plot,
            y_first_predict
        ],
        types: {
            true_data: "scatter",
            y_predict_1: "line"
        }
    },
    axis: {
        x: {
            label: {
                text: "X",
                position: "outer-middle"
            },
            tick: {
                fit: false
            }
        }
    }
});

var first_predict_coefficient = document.getElementById('first_predict_coefficient');
first_predict_coefficient.innerHTML = `<b>y = ${a.get()}x^3 + ${b.get()}x^2 + ${c.get()}x + ${d.get()}</b>`;

//損失関数の定義

function loss(predictions, labels) {
    //predictions, labelsは配列
    const meanSquareError = predictions.sub(labels).square().mean();
    return meanSquareError;
}

function train (xs, ys, numIterations = 75) {
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    for (let iter = 0; iter < numIterations; iter ++ ){
        //繰り返しの回数だけminimize関数が呼ばれる。Where th magic happens!
        optimizer.minimize(()=>{
            const predsYs = predict(xs);
            var output_loss = loss(predsYs, ys);
            loss_val.push(output_loss.get());
            return output_loss;
        });
    };
}

var loss_val = [];

train(x_val, y_val);

var debug3 = document.getElementById('debug3');
debug3.innerHTML = y_val;

//損失関数の値がどのように変化したかを見る。


function start_train() {
    console.log('push button');
    train(x_val, y_val);

    // ここに損失関数の値の結果を出力する文字列を入れていく
    var loss_val_text = "";

    for (var i = 0; i < loss_val.length; i++){
        loss_val_text += `<li>${i+1}回目：${loss_val[i]}</li>`
    };

    //結果の出力
    var loss_val_output_html = document.getElementById('loss_val');
    loss_val_output_html.innerHTML = "<ul>" + loss_val_text + "</ul>";
}




// ここに損失関数の値の結果を出力する文字列を入れていく
var loss_val_text = "";

for (var i = 0; i < loss_val.length; i++){
    loss_val_text += `<li>${i+1}回目：${loss_val[i]}</li>`
};

//結果の出力
var loss_val_output_html = document.getElementById('loss_val');
loss_val_output_html.innerHTML = "<ul>" + loss_val_text + "</ul>" + "hoge";
