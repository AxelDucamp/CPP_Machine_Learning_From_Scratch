#include <iostream>
#include <vector>

using namespace std;

class MultiVariateLinearRegression
{

public:
    vector<float> W = {};
    vector<float> B = {};

    vector<float> grad_w = {};
    vector<float> grad_b = {};

    float loss = 0;
    float *ptr_loss = &loss;

    float pred = 0;
    float *ptr_pred = &pred;

    void init(int size)
    {
        for (int i = 0; i <= size - 1; i++)
        {
            W.push_back(0.001);
            grad_w.push_back(0.0);
        }

        B.push_back(0.0);
        grad_b.push_back(0.0);
    }

    float compute_loss(float y_true, float y_pred)
    {
        float loss = 0;
        float *ptr_l = &loss;

        *ptr_l = (y_true - y_pred) * (y_true - y_pred);
        return *ptr_l;
    }

    void predict(vector<vector<float>> value)
    {

        float pred_val = 0;
        float *ptr_pred_val = &pred_val;

        for (int i = 0; i <= value.size() - 1; i++)
        {
            for (int j = 0; j <= value[i].size() - 1; j++)
            {
                *ptr_pred_val += W[j] * value[i][j];
            }

            *ptr_pred_val += B[0];

            cout << "Predicted Value : " << *ptr_pred_val << endl;

            *ptr_pred_val = 0;
        }
    }

    void fit(vector<vector<float>> value, vector<float> true_labels, int epochs, float learning_rate = 0.01)
    {
        init(value[0].size());
        cout << "=======================================================" << endl;
        for (int epo = 0; epo <= epochs - 1; epo++)
        {
            for (int i = 0; i <= value.size() - 1; i++)
            {
                for (int j = 0; j <= value[i].size() - 1; j++)
                {
                    *ptr_pred += W[j] * value[i][j];
                }

                *ptr_pred += B[0];

                *ptr_loss += compute_loss(true_labels[i], *ptr_pred);

                for (int j = 0; j <= value[i].size() - 1; j++)
                {
                    grad_w[j] = -2 * (true_labels[i] - *ptr_pred) * value[i][j];

                    W[j] = W[j] - learning_rate * grad_w[j];

                    grad_b[0] = -2 * (true_labels[i] - *ptr_pred) * 1;

                    B[0] = B[0] - learning_rate * grad_b[j];
                }

                *ptr_pred = 0;
            }

            *ptr_loss = *ptr_loss / value.size();

            cout << "Epoch : " << epo + 1 << " Loss : " << *ptr_loss << endl;
            cout << "=======================================================" << endl;
            *ptr_loss = 0;
        }
    }

    void evaluate(vector<vector<float>> value, vector<float> true_labels)
    {
        float pred_val_eval = 0;
        float *ptr_pred_val_eval = &pred_val_eval;

        float loss_eval = 0;
        float *ptr_loss_eval = &loss_eval;

        for (int i = 0; i <= value.size() - 1; i++)
        {
            for (int j = 0; j <= value[i].size() - 1; j++)
            {
                *ptr_pred_val_eval += W[j] * value[i][j];
            }

            *ptr_pred_val_eval += B[0];

            *ptr_loss_eval += compute_loss(true_labels[i], *ptr_pred_val_eval);

            *ptr_pred_val_eval = 0;
        }

        cout << "\nModel MSE : " << *ptr_loss_eval / value.size() << endl;
    }
};

int main()
{

    MultiVariateLinearRegression lm;

    vector<vector<float>> value = {{1.}, {2.}, {3.}};
    vector<float> true_labels = {2, 3, 4};
    int epochs = 100;
    float learning_rate = 0.1;

    lm.fit(value, true_labels, epochs, learning_rate);

    vector<vector<float>> value_to_pred = {{4.}, {5.}, {6.}, {50.}};

    lm.predict(value_to_pred);

    vector<vector<float>> value_eval = {{10.}, {20.}, {30.}};
    vector<float> true_labels_eval = {11., 21., 31.};

    lm.evaluate(value_eval, true_labels_eval);

    return 0;
}
