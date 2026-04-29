


(base) [agentrogue@LAAPY Vitas_ai]$ 
 *  History restored 

(base) [agentrogue@LAAPY Vitas_ai]$ export HF_HUB_ENABLE_HF_TRANSFER=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib/
~/miniconda3/bin/python train.py
Loading weights: 100%|██████████████████████████████████████████████████████████████| 291/291 [00:00<00:00, 998.82it/s]
Adding EOS to train dataset: 100%|███████████████████████████████████████| 7828/7828 [00:00<00:00, 35302.26 examples/s]
Tokenizing train dataset: 100%|███████████████████████████████████████████| 7828/7828 [00:01<00:00, 4529.50 examples/s]
Truncating train dataset: 100%|█████████████████████████████████████████| 7828/7828 [00:00<00:00, 594574.74 examples/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 32000}.
{'loss': '2.55', 'grad_norm': '0.1436', 'learning_rate': '0', 'entropy': '1.85', 'num_tokens': '896', 'mean_token_accuracy': '0.5463', 'epoch': '0.001022'}
{'loss': '2.54', 'grad_norm': '0.1572', 'learning_rate': '4e-05', 'entropy': '1.908', 'num_tokens': '1721', 'mean_token_accuracy': '0.5345', 'epoch': '0.002044'}
{'loss': '2.111', 'grad_norm': '0.1027', 'learning_rate': '8e-05', 'entropy': '1.737', 'num_tokens': '2829', 'mean_token_accuracy': '0.5547', 'epoch': '0.003066'}
{'loss': '2.261', 'grad_norm': '0.1248', 'learning_rate': '0.00012', 'entropy': '1.788', 'num_tokens': '3743', 'mean_token_accuracy': '0.5588', 'epoch': '0.004088'}
{'loss': '1.942', 'grad_norm': '0.1029', 'learning_rate': '0.00016', 'entropy': '1.673', 'num_tokens': '5041', 'mean_token_accuracy': '0.585', 'epoch': '0.00511'}
{'loss': '2.17', 'grad_norm': '0.1469', 'learning_rate': '0.0002', 'entropy': '1.932', 'num_tokens': '6040', 'mean_token_accuracy': '0.5332', 'epoch': '0.006132'}
{'loss': '2.146', 'grad_norm': '0.2131', 'learning_rate': '0.0001964', 'entropy': '1.918', 'num_tokens': '6858', 'mean_token_accuracy': '0.5776', 'epoch': '0.007154'}
{'loss': '1.742', 'grad_norm': '0.1326', 'learning_rate': '0.0001927', 'entropy': '1.629', 'num_tokens': '7906', 'mean_token_accuracy': '0.6108', 'epoch': '0.008176'}
{'loss': '1.813', 'grad_norm': '0.3422', 'learning_rate': '0.0001891', 'entropy': '1.685', 'num_tokens': '8833', 'mean_token_accuracy': '0.5966', 'epoch': '0.009198'}
{'loss': '1.593', 'grad_norm': '0.246', 'learning_rate': '0.0001855', 'entropy': '1.558', 'num_tokens': '9939', 'mean_token_accuracy': '0.6382', 'epoch': '0.01022'}
{'loss': '1.577', 'grad_norm': '0.1393', 'learning_rate': '0.0001818', 'entropy': '1.581', 'num_tokens': '1.121e+04', 'mean_token_accuracy': '0.657', 'epoch': '0.01124'}
{'loss': '1.437', 'grad_norm': '0.1808', 'learning_rate': '0.0001782', 'entropy': '1.525', 'num_tokens': '1.223e+04', 'mean_token_accuracy': '0.6926', 'epoch': '0.01226'}
{'loss': '1.306', 'grad_norm': '0.2299', 'learning_rate': '0.0001745', 'entropy': '1.406', 'num_tokens': '1.315e+04', 'mean_token_accuracy': '0.7113', 'epoch': '0.01329'}
{'loss': '1.363', 'grad_norm': '0.2326', 'learning_rate': '0.0001709', 'entropy': '1.434', 'num_tokens': '1.43e+04', 'mean_token_accuracy': '0.6906', 'epoch': '0.01431'}
{'loss': '1.335', 'grad_norm': '0.2902', 'learning_rate': '0.0001673', 'entropy': '1.407', 'num_tokens': '1.535e+04', 'mean_token_accuracy': '0.707', 'epoch': '0.01533'}
{'loss': '1.236', 'grad_norm': '0.09065', 'learning_rate': '0.0001636', 'entropy': '1.257', 'num_tokens': '1.636e+04', 'mean_token_accuracy': '0.7391', 'epoch': '0.01635'}
{'loss': '1.352', 'grad_norm': '0.08666', 'learning_rate': '0.00016', 'entropy': '1.429', 'num_tokens': '1.782e+04', 'mean_token_accuracy': '0.7059', 'epoch': '0.01737'}
{'loss': '1.143', 'grad_norm': '0.08128', 'learning_rate': '0.0001564', 'entropy': '1.115', 'num_tokens': '1.869e+04', 'mean_token_accuracy': '0.7589', 'epoch': '0.0184'}
{'loss': '1.147', 'grad_norm': '0.06237', 'learning_rate': '0.0001527', 'entropy': '1.187', 'num_tokens': '1.967e+04', 'mean_token_accuracy': '0.7477', 'epoch': '0.01942'}
{'loss': '1.051', 'grad_norm': '0.06567', 'learning_rate': '0.0001491', 'entropy': '1.125', 'num_tokens': '2.06e+04', 'mean_token_accuracy': '0.7566', 'epoch': '0.02044'}
{'loss': '1.043', 'grad_norm': '0.06135', 'learning_rate': '0.0001455', 'entropy': '1.122', 'num_tokens': '2.14e+04', 'mean_token_accuracy': '0.7676', 'epoch': '0.02146'}
{'loss': '0.9837', 'grad_norm': '0.06841', 'learning_rate': '0.0001418', 'entropy': '1.042', 'num_tokens': '2.219e+04', 'mean_token_accuracy': '0.7892', 'epoch': '0.02248'}
{'loss': '1.138', 'grad_norm': '0.07639', 'learning_rate': '0.0001382', 'entropy': '1.127', 'num_tokens': '2.329e+04', 'mean_token_accuracy': '0.7409', 'epoch': '0.02351'}
{'loss': '1.047', 'grad_norm': '0.07866', 'learning_rate': '0.0001345', 'entropy': '1.066', 'num_tokens': '2.402e+04', 'mean_token_accuracy': '0.7737', 'epoch': '0.02453'}
{'loss': '1.161', 'grad_norm': '0.06754', 'learning_rate': '0.0001309', 'entropy': '1.222', 'num_tokens': '2.517e+04', 'mean_token_accuracy': '0.7331', 'epoch': '0.02555'}
{'loss': '1.152', 'grad_norm': '0.08194', 'learning_rate': '0.0001273', 'entropy': '1.188', 'num_tokens': '2.615e+04', 'mean_token_accuracy': '0.7412', 'epoch': '0.02657'}
{'loss': '1.18', 'grad_norm': '0.06312', 'learning_rate': '0.0001236', 'entropy': '1.187', 'num_tokens': '2.705e+04', 'mean_token_accuracy': '0.7391', 'epoch': '0.02759'}
{'loss': '0.9249', 'grad_norm': '0.0615', 'learning_rate': '0.00012', 'entropy': '0.9868', 'num_tokens': '2.784e+04', 'mean_token_accuracy': '0.7979', 'epoch': '0.02862'}
{'loss': '0.9293', 'grad_norm': '0.06436', 'learning_rate': '0.0001164', 'entropy': '0.9439', 'num_tokens': '2.873e+04', 'mean_token_accuracy': '0.7863', 'epoch': '0.02964'}
{'loss': '1.19', 'grad_norm': '0.07703', 'learning_rate': '0.0001127', 'entropy': '1.173', 'num_tokens': '2.993e+04', 'mean_token_accuracy': '0.7354', 'epoch': '0.03066'}
{'loss': '1.209', 'grad_norm': '0.07699', 'learning_rate': '0.0001091', 'entropy': '1.221', 'num_tokens': '3.107e+04', 'mean_token_accuracy': '0.7334', 'epoch': '0.03168'}
{'loss': '1.185', 'grad_norm': '0.07441', 'learning_rate': '0.0001055', 'entropy': '1.061', 'num_tokens': '3.186e+04', 'mean_token_accuracy': '0.7581', 'epoch': '0.0327'}
{'loss': '1.201', 'grad_norm': '0.06606', 'learning_rate': '0.0001018', 'entropy': '1.067', 'num_tokens': '3.302e+04', 'mean_token_accuracy': '0.7543', 'epoch': '0.03373'}
{'loss': '1.205', 'grad_norm': '0.06009', 'learning_rate': '9.818e-05', 'entropy': '1.108', 'num_tokens': '3.398e+04', 'mean_token_accuracy': '0.7408', 'epoch': '0.03475'}
{'loss': '1.094', 'grad_norm': '0.07202', 'learning_rate': '9.455e-05', 'entropy': '1.089', 'num_tokens': '3.49e+04', 'mean_token_accuracy': '0.7715', 'epoch': '0.03577'}
{'loss': '1.154', 'grad_norm': '0.07142', 'learning_rate': '9.091e-05', 'entropy': '1.079', 'num_tokens': '3.581e+04', 'mean_token_accuracy': '0.7546', 'epoch': '0.03679'}
{'loss': '1.232', 'grad_norm': '0.07564', 'learning_rate': '8.727e-05', 'entropy': '1.136', 'num_tokens': '3.671e+04', 'mean_token_accuracy': '0.7495', 'epoch': '0.03781'}
{'loss': '0.991', 'grad_norm': '0.06592', 'learning_rate': '8.364e-05', 'entropy': '0.9812', 'num_tokens': '3.778e+04', 'mean_token_accuracy': '0.7817', 'epoch': '0.03883'}
{'loss': '1.135', 'grad_norm': '0.06818', 'learning_rate': '8e-05', 'entropy': '1.182', 'num_tokens': '3.883e+04', 'mean_token_accuracy': '0.754', 'epoch': '0.03986'}
{'loss': '1.184', 'grad_norm': '0.06787', 'learning_rate': '7.636e-05', 'entropy': '1.098', 'num_tokens': '3.978e+04', 'mean_token_accuracy': '0.7367', 'epoch': '0.04088'}
{'loss': '1.294', 'grad_norm': '0.06276', 'learning_rate': '7.273e-05', 'entropy': '1.293', 'num_tokens': '4.115e+04', 'mean_token_accuracy': '0.727', 'epoch': '0.0419'}
{'loss': '1.061', 'grad_norm': '0.06303', 'learning_rate': '6.909e-05', 'entropy': '1.062', 'num_tokens': '4.209e+04', 'mean_token_accuracy': '0.765', 'epoch': '0.04292'}
{'loss': '1.089', 'grad_norm': '0.07029', 'learning_rate': '6.545e-05', 'entropy': '1.136', 'num_tokens': '4.323e+04', 'mean_token_accuracy': '0.7548', 'epoch': '0.04394'}
{'loss': '1.1', 'grad_norm': '0.0711', 'learning_rate': '6.182e-05', 'entropy': '1.144', 'num_tokens': '4.418e+04', 'mean_token_accuracy': '0.7567', 'epoch': '0.04497'}
{'loss': '1.274', 'grad_norm': '0.08429', 'learning_rate': '5.818e-05', 'entropy': '1.187', 'num_tokens': '4.51e+04', 'mean_token_accuracy': '0.7354', 'epoch': '0.04599'}
{'loss': '1.251', 'grad_norm': '0.06144', 'learning_rate': '5.455e-05', 'entropy': '1.211', 'num_tokens': '4.632e+04', 'mean_token_accuracy': '0.7161', 'epoch': '0.04701'}
{'loss': '1.143', 'grad_norm': '0.07052', 'learning_rate': '5.091e-05', 'entropy': '1.18', 'num_tokens': '4.776e+04', 'mean_token_accuracy': '0.7376', 'epoch': '0.04803'}
{'loss': '1.14', 'grad_norm': '0.05493', 'learning_rate': '4.727e-05', 'entropy': '1.181', 'num_tokens': '4.873e+04', 'mean_token_accuracy': '0.7587', 'epoch': '0.04905'}
{'loss': '1.002', 'grad_norm': '0.0637', 'learning_rate': '4.364e-05', 'entropy': '0.9819', 'num_tokens': '4.953e+04', 'mean_token_accuracy': '0.7896', 'epoch': '0.05008'}
{'loss': '1.065', 'grad_norm': '0.05858', 'learning_rate': '4e-05', 'entropy': '1.117', 'num_tokens': '5.049e+04', 'mean_token_accuracy': '0.7588', 'epoch': '0.0511'}
{'loss': '1.264', 'grad_norm': '0.07634', 'learning_rate': '3.636e-05', 'entropy': '1.115', 'num_tokens': '5.141e+04', 'mean_token_accuracy': '0.7298', 'epoch': '0.05212'}
{'loss': '1.071', 'grad_norm': '0.05962', 'learning_rate': '3.273e-05', 'entropy': '1.047', 'num_tokens': '5.233e+04', 'mean_token_accuracy': '0.7623', 'epoch': '0.05314'}
{'loss': '1.254', 'grad_norm': '0.07294', 'learning_rate': '2.909e-05', 'entropy': '1.232', 'num_tokens': '5.33e+04', 'mean_token_accuracy': '0.7165', 'epoch': '0.05416'}
{'loss': '0.8863', 'grad_norm': '0.07378', 'learning_rate': '2.545e-05', 'entropy': '1.022', 'num_tokens': '5.416e+04', 'mean_token_accuracy': '0.7997', 'epoch': '0.05519'}
{'loss': '1.309', 'grad_norm': '0.06449', 'learning_rate': '2.182e-05', 'entropy': '1.212', 'num_tokens': '5.516e+04', 'mean_token_accuracy': '0.7145', 'epoch': '0.05621'}
{'loss': '1.154', 'grad_norm': '0.06337', 'learning_rate': '1.818e-05', 'entropy': '1.151', 'num_tokens': '5.604e+04', 'mean_token_accuracy': '0.7594', 'epoch': '0.05723'}
{'loss': '0.9704', 'grad_norm': '0.06121', 'learning_rate': '1.455e-05', 'entropy': '1.016', 'num_tokens': '5.709e+04', 'mean_token_accuracy': '0.782', 'epoch': '0.05825'}
{'loss': '1.063', 'grad_norm': '0.06898', 'learning_rate': '1.091e-05', 'entropy': '1.063', 'num_tokens': '5.807e+04', 'mean_token_accuracy': '0.7707', 'epoch': '0.05927'}
{'loss': '0.9701', 'grad_norm': '0.06714', 'learning_rate': '7.273e-06', 'entropy': '1.028', 'num_tokens': '5.891e+04', 'mean_token_accuracy': '0.7728', 'epoch': '0.0603'}
{'loss': '1.281', 'grad_norm': '0.07186', 'learning_rate': '3.636e-06', 'entropy': '1.335', 'num_tokens': '6.002e+04', 'mean_token_accuracy': '0.7259', 'epoch': '0.06132'}
{'train_runtime': '178', 'train_samples_per_second': '2.697', 'train_steps_per_second': '0.337', 'train_loss': '1.313', 'epoch': '0.06132'}
100%|██████████████████████████████████████████████████████████████████████████████████| 60/60 [02:57<00:00,  2.97s/it]
Training complete! Model saved to 'ayurveda_lora_model'
Exception ignored in: <_io.BufferedWriter name=28>
BrokenPipeError: [Errno 32] Broken pipe


✅ All done! Results saved to './eval_results/'
   • ./eval_results/training_dashboard.png
   • ./eval_results/inference_metrics.png
   • ./eval_results/metrics.json

📊 Final Metrics:
   ROUGE-1 : 0.4257
   ROUGE-2 : 0.2116
   ROUGE-L : 0.3379
   PPL Mean: 9.49