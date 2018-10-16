
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
debug.innerHTML = "Debug::  " + "y_val : " + y_val_with_noise.length;

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
debug2.innerHTML = "Debug2:: " + "y_plot length : " + y_plot.length + "  x_plot length : " + x_plot.length;

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

var y_first_predict = ["y_predict_1"];

// 予想の曲線を表示している
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
        },
        y: {
            label: {
                text: "Y",
                position: "outer-middle"
            }
        }
    }
});

var first_predict_coefficient = document.getElementById('first_predict_coefficient');
first_predict_coefficient.innerHTML = `<b>y = ${a_result}x<sup>3</sup> + ${b_result}x<sup>2</sup> + ${c_result}x + ${d_result}</b>`;

//損失関数の定義

function loss(predictions, labels) {
    //predictions, labelsは配列
    const meanSquareError = predictions.sub(labels).square().mean();
    return meanSquareError;
}

function train (xs, ys, numIterations = 100) {
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    for (let iter = 0; iter < numIterations; iter ++ ){
        //繰り返しの回数だけminimize関数が呼ばれる。Where th magic happens!
        optimizer.minimize(()=>{
            const predsYs = predict(xs);
            var output_loss = loss(predsYs, ys);
            loss_val.push(output_loss.get()); //listに損失関数の値を入れていく
            return output_loss;
        });
    };
}


var debug3 = document.getElementById('debug3');
debug3.innerHTML = "Debug3:: " + "y_val : " + y_val;

//損失関数の値がどのように変化したかを見る。

var loss_val = []; //損失関数の値を追加していくためのリスト

// ここに損失関数の値の結果を出力する文字列を入れていく
var loss_val_text = "";


function start_train() {

    train(x_val, y_val);


    for (var i = 0; i < loss_val.length; i++){
        loss_val_text += `<li>${i+1}回目：${loss_val[i]}</li>`
    };

    //結果の出力
    var loss_val_text_html = document.getElementById('loss_val_text_html');
    loss_val_text_html.innerHTML = "<b>学習回数と損失関数の値</b>";
    var loss_val_output_html = document.getElementById('loss_val');
    loss_val_output_html.innerHTML = "<ul>" + loss_val_text + "</ul>";

    //学習開始のボタンを消す
    document.getElementById('train_button').style.display = "none";

    const number_of_x_loss_plot = loss_val.length + 1;

    var x_loss_plot_tensor = tf.range(1.0, number_of_x_loss_plot, 1.0);
    var x_loss_plot = tensor1d_to_array(x_loss_plot_tensor);

    //損失関数の変化を図示するためにx, yにラベルづけ
    loss_val.unshift('loss_plot');
    x_loss_plot.unshift('X');

    var debug4 = document.getElementById("debug4");
    debug4.innerHTML = "Debug4:: " + `x : ${x_loss_plot}, y : ${loss_val}`;

    //損失関数の値を図示

    var loss_plot_text = document.getElementById("loss_plot_text");
    loss_plot_text.innerHTML = "<b>損失関数の値の変化</b>";

    var loss_plot = c3.generate({
        bindto: "#loss_plot",
        size: {
            height: 500,
            width: 650
        },
        data: {
            xs: {
                loss_plot: "X"
            },
            columns: [
                x_loss_plot,
                loss_val
            ]
        },
        axis: {
            x: {
                label: {
                    text: "学習回数",
                    position: "outer-middle"
                },
                tick: {
                    fit: false
                }
            },
            y: {
                label: {
                    text: "損失関数の値",
                    position: "outer-middle"
                }
            }
        }
    });
    
    //フィットさせた結果を図示するために配列を作ってる
    var y_result_predict = ["y_result_predict"];

    x_val_array.forEach((x) => {
        return tf.tidy(()=>{
            var x_tf = tf.scalar(x);
            y_result_predict.push(predict(x_tf).asScalar().get());
        });    
    });

    //結果を図示する

    var fit_function_text = document.getElementById('fit_function_text');
    fit_function_text.innerHTML = `<b>係数を学習してフィットさせた図</b><br><b>y = ${a.get()}x<sup>3</sup> + ${b.get()}x<sup>2</sup> + ${c.get()}x + ${d.get()}</b>`

    var debug5 = document.getElementById('debug5');
    debug5.innerHTML = "Debug5::  " + `x_plot : ${x_plot.length}, y_plot : ${y_plot.length}, y_result_pred : ${y_result_predict.length}`;
    var fit_function = c3.generate({
        bindto: "#fit_function",
        size: {
            height: 500,
            width: 500
        },
        data: {
            xs: {
                true_data: "t_data_x",
                y_result_predict: "t_data_x"
            },
            columns: [
                x_plot,
                y_plot,
                y_result_predict
            ],
            types: {
                true_data: "scatter",
                y_result_predict: "line"
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
            },
            y: {
                label: {
                    text: "Y",
                    position: "outer-middle"
                }
            }
        }
    });

    var final_result = document.getElementById('final_result');
    final_result.innerHTML = `<b>
        <h2>最終結果</h2>
        <p>設定した値： a=-0.80, b=-0.20, c=0.90, d=0.50<p>
        <p>学習した値：a=${round_2(a.get())}, b=${round_2(b.get())}, c=${round_2(c.get())}, d=${round_2(d.get())}</p>
    </b>`;
}

function round_2 (val){
    return Math.round(val*100)/100;
}
