/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/exec_code.py:121 in exec_func_with_error_handling                        
                                                                                
  /home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:640 in code_to_exec                                     
                                                                                
  /mount/src/term-project-csis-4260/streamlit_app.py:124 in <module>            
                                                                                
    121 │   df = load_data()                                                    
    122 │   model_path, scaler_X_path, scaler_y_path, lookback, scale_y = mode  
    123 │   model = load_model(model_path)                                      
  ❱ 124 │   scaler_X = joblib.load(scaler_X_path)                               
    125 │   scaler_y = joblib.load(scaler_y_path) if scale_y else None          
    126 │   X = df[required_features]                                           
    127 │   future_dates, future_preds = predict_next_days(model, scaler_X, sc  
                                                                                
  /home/adminuser/venv/lib/python3.12/site-packages/joblib/numpy_pickle.py:650  
  in load                                                                       
                                                                                
    647 │   │   with _read_fileobject(fobj, filename, mmap_mode) as fobj:       
    648 │   │   │   obj = _unpickle(fobj)                                       
    649 │   else:                                                               
  ❱ 650 │   │   with open(filename, 'rb') as f:                                 
    651 │   │   │   with _read_fileobject(f, filename, mmap_mode) as fobj:      
    652 │   │   │   │   if isinstance(fobj, str):                               
    653 │   │   │   │   │   # if the returned file object is a string, this me  
────────────────────────────────────────────────────────────────────────────────
FileNotFoundError: [Errno 2] No such file or directory: 
'models/scaler_X_7day.pkl'
