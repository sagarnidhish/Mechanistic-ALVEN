import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#import nonlinear_regression as nr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
import math
import numpy.matlib as matlib
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



def numerical_derivative(x, t):
    return np.gradient(x, t)

def ALVEN_features(X, y, degree):

    def _xexp(x):
        '''exponential transform with protection against large numbers'''
        with np.errstate(over='ignore'):
            return np.where(np.abs(x) < 9, np.exp(x), np.exp(9)*np.ones_like(x))

    def _xlog(x):
        '''logarithm with protection agiasnt small numbers'''
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            return np.where(np.abs(x) > np.exp(-10), np.log(abs(x)), -10*np.ones_like(x))
 
    def _xsqrt(x):
        '''square root with protection with negative values (take their abs)'''
        with np.errstate(invalid = 'ignore'):
            return np.sqrt(np.abs(x))
    
    def _xinv(x):
        '''inverse with protection with 0 value'''
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            return np.where(np.abs(x)>1e-9, 1/x, 1e9*np.ones_like(x))
        
    Xlog = _xlog(X)
    Xinv = _xinv(X)    
    Xsqrt = _xsqrt(X)
    Xexp = _xexp(X)

    #if X.ndim == 1:
    #    X = X.reshape(-1, 1)

    #print(X.shape)

    if degree == 1:
        X = np.column_stack((X, Xlog, Xsqrt, Xinv))
    
    #print(X.shape)

    #if degree == 2:
    #    X = np.column_stack((X, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv))

    #print(X.shape)

    if degree == 2:
        poly = PolynomialFeatures(degree = 2,include_bias=False, interaction_only = True)
        X_inter = poly.fit_transform(X)[:,X.shape[1]:]
        print(X_inter.shape)
        X = np.column_stack((X, X_inter, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv)) # 11 values, stored in 0 to 10th columns
        print(X.shape)
    
    #if degree == 3:
    #    X = np.column_stack((X, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv, X**3, Xlog**3, Xinv**3, X**2*Xsqrt, Xlog**2*Xinv, Xlog*Xsqrt*Xinv,Xlog*Xinv**2, Xsqrt*Xinv**2))

    if degree == 3:
         poly = PolynomialFeatures(degree = 3,include_bias=False, interaction_only = True)
         X_inter = poly.fit_transform(X)[:,X.shape[1]:]
            
         X = np.column_stack((X, X_inter, Xlog, Xsqrt, Xinv, X**2,Xlog**2,Xinv**2, X*Xsqrt, Xlog*Xinv, Xsqrt*Xinv, X**3, Xlog**3, Xinv**3, X**2*Xsqrt, Xlog**2*Xinv, Xlog*Xsqrt*Xinv,Xlog*Xinv**2, Xsqrt*Xinv**2))

    # You may need to remove features with 0 variance and standardize the data

    # Remove feature with 0 variance
    sel = VarianceThreshold(threshold=tol).fit(X)
    X = sel.transform(X)
    X_gen_orig = X

    # Z-score data
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X = scaler_x.transform(X)

    y_orig = y
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    y = y.reshape(-1, 1)
    scaler_y.fit(y)
    y = scaler_y.transform(y)

    #eliminate feature
    f_test, p_values = f_regression(X, y.flatten())

    # Display the F-statistics and p-values
    print("F-statistics:", f_test)
    print("p-values:", p_values)

    if selection == 'p_value':
        X_fit = X[:,p_values<select_value]
        #X_test_fit = X_test[:,p_values<select_value]
        retain_index = p_values<select_value
        
    elif selection == 'percentage':
        number = int(math.ceil(select_value * X.shape[1]))
        f_test.sort()
        value = f_test[-number]
        X_fit =  X[:,f_test>=value]
        #X_test_fit = X_test[:,f_test>=value]
        
        retain_index = f_test>=value
        
    else:
        f = np.copy(f_test)
        f.sort()  #descending order
        f = f[::-1]
        
        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1),f.reshape(-1,1)),axis=1)
        
        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        
        #find the distance from each point to the line
        vecFromFirst = AllCord- AllCord[0]
        #and calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        
        X_fit =  X[:,f_test>=value]
        #X_test_fit = X_test[:,f_test>=value]        
        
        retain_index = f_test>=value # the retained features


    #choose the appropriate alpha in cross_Validation: cv= True

    if X_fit.shape[1] == 0:
        print('no variable selected by ALVEN')
        ALVEN_model = None
        ALVEN_params = None
        mse_train = np.var(pH_meas)
        #mse_test = np.var(y_test)
        yhat_train = np.zeros(pH_meas.shape)
        #yhat_test = np.zeros(y_test.shape)
        global alpha
        alpha = 0
    else: 
        if alpha_num is not None and cv:
            #X_max = np.concatenate((X_fit,X_test_fit),axis = 0)
            #y_max = np.concatenate((y, y_test), axis = 0)
            X_max = X_fit
            y_max = pH_meas
            alpha_max = (np.sqrt(np.sum(np.dot(X_max.T,y_max) ** 2, axis=0)).max())/X_max.shape[0]/l1_ratio 
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1] # Generating alpha values between log(alpha_max*tol) and log(alpha_max)
            # The last [::-1] is to reverse the order of the list, so that the values are sorted in descending order
            alpha = alpha_list[alpha]
        
        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(X_fit.T,y) ** 2, axis=0)).max())/X_fit.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]


    return X_fit, y, X_gen_orig, y_orig

def custom_objective(params, X, y, alpha, l1_ratio, alpha_mech, beta_mech, t, p_mech, u, d, Ybase, Yacid):
    # Fit ElasticNet model
    #model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
    #model.coef_ = params
    #model._validate_params()
    
    
    predictions = X.dot(params)
    residuals = predictions - y

    # elasticnet_loss = np.mean(residuals**2) + model._get_penalty()

    # Compute ElasticNet loss
    mse_loss = np.mean(residuals**2)  # check if this correct
    l1_penalty = l1_ratio * np.sum(np.abs(params))
    l2_penalty = ((1 - l1_ratio)/2) * np.sum(params**2)
    elasticnet_loss = (mse_loss + alpha * (l1_penalty + l2_penalty)) / (2 * len(y))
    
    #Compute mechanistic equality penalty
    h = p_mech * numerical_derivative(X[:, 0], t) - u * (Ybase - X[:, 0]) - d * (Yacid - X[:, 0])
    mech_eq_penalty = alpha_mech * np.linalg.norm(h, ord=2) / (2 * len(h))
    #print("Mech eq penalty:", mech_eq_penalty)
    
    #Compute mechanistic inequality penalty
    dot_product = X.dot(params) # substitute this with predictions variable
    g = -numerical_derivative(dot_product, X[:, 0])
    g = -1 * numerical_derivative(dot_product, X[:, 0])
    mech_ineq_penalty = beta_mech * np.linalg.norm(np.maximum(0, g), ord=2) / (2 * len(g))
    #print("Mech ineq penalty:", mech_ineq_penalty)  
    
    # Total loss
    total_loss = elasticnet_loss + mech_eq_penalty + mech_ineq_penalty
    print("total_loss: ", total_loss)
    return total_loss

def fit_custom_elastic_net(X, y, alpha, l1_ratio, alpha_mech, beta_mech, t, p_mech, u, d, Ybase, Yacid):
    #initial_params = np.zeros(X.shape[1])
    p_ml = np.full(X.shape[1], 0.5)
    p_mech = V

    # Define options for the optimizer
    options = {
        'maxiter': 5000,  
        'disp': True    
    }
    result = minimize(
        custom_objective, 
        p_ml, 
        args=(X, y, alpha, l1_ratio, alpha_mech, beta_mech, t, p_mech, u, d, Ybase, Yacid),
        method='L-BFGS-B',
        #method='Nelder-Mead',
        tol=1e-4,         
        options=options   
    )
    
    if result.success:
        print("result.x: ", result.x)
        return result.x
    else:
        raise ValueError("Optimization failed")


#def EN_fitting(Y, pH, Y_test, pH_test, alpha, l1_ratio, max_iter=10000, tol=1e-4):
    EN_model = ElasticNet(random_state=0, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=max_iter, tol=tol)
    EN_model.fit(Y, pH)
    EN_params = EN_model.coef_.reshape((-1, 1))
    pHhat_train = EN_model.predict(Y).reshape((-1, 1))
    mse_train = mean_squared_error(pH, pHhat_train)
    pHhat_test = EN_model.predict(Y_test).reshape((-1, 1))
    mse_test = mean_squared_error(pH_test, pHhat_test)
    return (EN_model, EN_params, mse_train, mse_test, pHhat_train, pHhat_test)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def EN_fitting(X_train, y_train, X_test, y_test, alpha, l1_ratio, max_iter=10000, tol=1e-4):
    # Initialize ElasticNet model with provided parameters
    EN_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=max_iter, tol=tol, random_state=0)
    
    # Fit the model to the training data
    EN_model.fit(X_train, y_train)
    
    # Get the coefficients (model parameters)
    EN_params = EN_model.coef_.reshape((-1, 1))
    
    # Predict on the training data
    y_pred_train = EN_model.predict(X_train).reshape((-1, 1))
    
    # Calculate the training mean squared error
    mse_train = mean_squared_error(y_train, y_pred_train)
    
    # Predict on the test data
    y_pred_test = EN_model.predict(X_test).reshape((-1, 1))
    
    # Calculate the test mean squared error
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    # Return the model, parameters, MSEs, and predictions
    return EN_model, EN_params, mse_train, mse_test, y_pred_train, y_pred_test


# Example data
u = 1  # mol/sec
d = 1  # mol/sec
V = 5  # L
Y_acid = 0.2  # mol/sec
Y_base = -0.2  # mol/sec
degree = 1 # Degree of the polynomial features
tol=1e-5
i = 0

alpha_mech = 0.5
beta_mech = 0.5
p_mech = V
selection = 'p_value'
select_value = 0.15


# Define parameters for ALVEN
alpha = 99  # Index of alpha in the alpha_list # Larger index here is choosing a smaller alpha value
l1_ratio = 0.5  # Mix between Lasso and Ridge
#degree = 3  # Degree of nonlinearity
alpha_num = 100  # Number of alphas to consider in cross-validation
cv = True  # Enable cross-validation


#learning_rate = 0.01
#num_iterations = 1000

# The below have to be determined through cross-validation, 
lambda_ = 0.5
#alpha = 0.01
#alpha = 0.5

file_path = '/Users/ns/Desktop/Lab/MALVEN/pH_data.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

# Extract the first column (excluding the header)
time_data = df.iloc[1:, 0]
#time_data = df.iloc[:, 0]
t = time_data.to_numpy()

x_data = df.iloc[1:, 1]
x = x_data.to_numpy()
x = x.reshape(-1, 1)
#x = np.linspace(1, 2, 5)

pH_meas_data = df.iloc[1:, 2]
pH_meas = pH_meas_data.to_numpy()
pH_meas = pH_meas.flatten() 
pH_meas = pH_meas.T
print("Shape of pH_meas", pH_meas.shape)
print("pH_meas:", pH_meas)
pH_meas_orig = pH_meas
#pH_meas = np.linspace(2, 12, 5)
pH_meas = pH_meas.reshape(-1, 1)
#t = np.linspace(1, 5, 5)

X_fit, y, X_gen_orig, y_orig = ALVEN_features(x, pH_meas, degree)
print("x_fit shape:", X_fit.shape)
print("X_fit", X_fit)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fit, y, test_size=0.2, random_state=42
)

# fit model
p_ml = fit_custom_elastic_net(
    X_train, y_train,
    alpha, l1_ratio,
    alpha_mech, beta_mech,
    t, p_mech, u, d,
    Y_base, Y_acid
)
print("Fitted parameters p_ml:", p_ml)

# scale outputs
scaler_y = StandardScaler(with_mean=True, with_std=True)
scaler_y.fit(y_train)

# predictions on training set
y_train_pred = (p_ml @ X_train.T).reshape(-1, 1)
y_train_pred = scaler_y.inverse_transform(y_train_pred)


# predictions on test set
y_test_pred = (p_ml @ X_test.T).reshape(-1, 1)
y_test_pred = scaler_y.inverse_transform(y_test_pred)

# evaluation
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test,  y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2  = r2_score(y_test,  y_test_pred)
print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Test  MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# plotting
plt.figure(figsize=(8,5))
plt.scatter(x, pH_meas, label='All data', alpha=0.3)

# training predictions (in original x-space)
x_train = x[np.isin(X_fit, X_train).all(axis=1)]
plt.scatter(x_train, y_train_pred, marker='o', facecolors='none', edgecolors='green',
            label='Train predictions')

# test predictions
x_test = x[np.isin(X_fit, X_test).all(axis=1)]
plt.scatter(x_test, y_test_pred, marker='x', color='red', label='Test predictions')
plt.xlabel('x'), plt.ylabel('pH')
plt.title('Elastic Net Predictions on Train vs Test sets')
plt.legend(), plt.show()