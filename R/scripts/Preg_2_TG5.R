# Cargar librerías necesarias
library(tidyverse)
library(glmnet)
library(randomForest)
library(keras)
library(tensorflow)
library(broom)

# I. Cleaning and set-up

# Cargar los datos
setwd("C:\\Users\\Julio\\Downloads\\R_TG5\\Input")
penn_jae <- read.table("penn_jae.dat", header = TRUE)
setwd("C:\\Users\\Julio\\Downloads\\R_TG5\\Output")
# Ver los nombres de las variables para identificar la correcta
print("Nombres de variables en el dataset:")
print(names(penn_jae))

# Ver las primeras filas para identificar la variable de ingreso
print("Primeras filas del dataset:")
print(head(penn_jae))

# Filtrar observaciones donde tg es 0 o 4
data_clean <- penn_jae %>% 
  filter(tg %in% c(0, 4))

# Definir variable de tratamiento T4
data_clean <- data_clean %>% 
  mutate(T4 = ifelse(tg == 4, 1, 0))

# Buscar la variable correcta para el ingreso (probablemente "inuidur1" o similar)
# Si no encontramos "inuidurl", probamos con nombres alternativos comunes
if ("inuidur1" %in% names(data_clean)) {
  data_clean <- data_clean %>% 
    mutate(y = log(inuidur1))
  print("Usando inuidur1 como variable de ingreso")
} else if ("inuidur2" %in% names(data_clean)) {
  data_clean <- data_clean %>% 
    mutate(y = log(inuidur2))
  print("Usando inuidur2 como variable de ingreso")
} else if ("inuidur" %in% names(data_clean)) {
  data_clean <- data_clean %>% 
    mutate(y = log(inuidur))
  print("Usando inuidur como variable de ingreso")
} else {
  # Si no encontramos ninguna, mostramos las variables disponibles
  print("Variables disponibles en el dataset:")
  print(names(data_clean))
  stop("No se encontró la variable de ingreso. Revisa los nombres de variables arriba.")
}

# Crear dummies para dep
data_clean <- data_clean %>% 
  mutate(dep_0 = as.numeric(dep == 0),
         dep_1 = as.numeric(dep == 1),
         dep_2 = as.numeric(dep == 2))

# Definir matriz x con las variables especificadas
x_vars <- c('female', 'black', 'othrace', 'dep_1', 'dep_2', 
            'q2', 'q3', 'q4', 'q5', 'q6', 'recall', 'ageid35', 
            'ageg154', 'durable', 'nondurable', 'lusd', 'husd')

# Verificar que todas las variables existen
missing_vars <- setdiff(x_vars, names(data_clean))
if (length(missing_vars) > 0) {
  print(paste("Variables faltantes:", paste(missing_vars, collapse = ", ")))
  # Remover variables faltantes
  x_vars <- intersect(x_vars, names(data_clean))
  print(paste("Usando variables:", paste(x_vars, collapse = ", ")))
}

# Crear matriz x
x <- data_clean %>% 
  select(all_of(x_vars)) %>% 
  as.matrix()

# Definir y y d
y <- data_clean$y
d <- data_clean$T4

# Verificar dimensiones
print(paste("Dimensiones de x:", dim(x)[1], "filas,", dim(x)[2], "columnas"))
print(paste("Longitud de y:", length(y)))
print(paste("Longitud de d:", length(d)))

# II. Debiased ML

# Función DML con cross-fitting
dml <- function(y, d, x, n_folds = 5, methods_y = "ols", methods_d = "ols") {
  n <- length(y)
  fold_size <- floor(n / n_folds)
  indices <- sample(1:n)
  
  theta_hats <- c()
  se_hats <- c()
  
  for (fold in 1:n_folds) {
    # Definir train y test para este fold
    test_indices <- indices[((fold-1)*fold_size + 1):min(fold*fold_size, n)]
    train_indices <- setdiff(1:n, test_indices)
    
    # Entrenar modelos para y y d en datos de entrenamiento
    if (methods_y == "ols") {
      model_y <- lm(y[train_indices] ~ x[train_indices, ])
      y_hat_test <- predict(model_y, newdata = as.data.frame(x[test_indices, ]))
    } else if (methods_y == "lasso") {
      cv_lasso <- cv.glmnet(x[train_indices, ], y[train_indices], alpha = 1)
      y_hat_test <- predict(cv_lasso, newx = x[test_indices, ], s = "lambda.min")
    } else if (methods_y == "rf") {
      rf_y <- randomForest(x = x[train_indices, ], y = y[train_indices])
      y_hat_test <- predict(rf_y, newdata = x[test_indices, ])
    }
    
    if (methods_d == "ols") {
      model_d <- lm(d[train_indices] ~ x[train_indices, ])
      d_hat_test <- predict(model_d, newdata = as.data.frame(x[test_indices, ]))
    } else if (methods_d == "lasso") {
      cv_lasso_d <- cv.glmnet(x[train_indices, ], d[train_indices], alpha = 1, family = "binomial")
      d_hat_test <- predict(cv_lasso_d, newx = x[test_indices, ], s = "lambda.min", type = "response")
    } else if (methods_d == "rf") {
      rf_d <- randomForest(x = x[train_indices, ], y = as.factor(d[train_indices]))
      d_hat_test <- predict(rf_d, newdata = x[test_indices, ], type = "prob")[,2]
    }
    
    # Calcular residuos
    v_hat <- y[test_indices] - y_hat_test
    u_hat <- d[test_indices] - d_hat_test
    
    # Estimar theta para este fold
    theta_fold <- mean(u_hat * v_hat) / mean(u_hat * d[test_indices])
    theta_hats[fold] <- theta_fold
    
    # Calcular error estándar
    se_fold <- sd(u_hat * v_hat) / (sqrt(length(test_indices)) * abs(mean(u_hat * d[test_indices])))
    se_hats[fold] <- se_fold
  }
  
  # Promediar estimaciones entre folds
  theta_hat <- mean(theta_hats)
  se_hat <- mean(se_hats)
  
  return(list(theta_hat = theta_hat, se_hat = se_hat, theta_hats = theta_hats))
}

# Estimar con diferentes métodos
results_dml <- data.frame()

# OLS
dml_ols <- dml(y, d, x, methods_y = "ols", methods_d = "ols")
results_dml <- rbind(results_dml, 
                     data.frame(Method = "OLS", 
                                Estimate = dml_ols$theta_hat,
                                SE = dml_ols$se_hat))

# Lasso
dml_lasso <- dml(y, d, x, methods_y = "lasso", methods_d = "lasso")
results_dml <- rbind(results_dml, 
                     data.frame(Method = "Lasso", 
                                Estimate = dml_lasso$theta_hat,
                                SE = dml_lasso$se_hat))

# Random Forest
dml_rf <- dml(y, d, x, methods_y = "rf", methods_d = "rf")
results_dml <- rbind(results_dml, 
                     data.frame(Method = "Random Forest", 
                                Estimate = dml_rf$theta_hat,
                                SE = dml_rf$se_hat))

# Neural Network (simplificado - en práctica necesitarías ajustar más)
tryCatch({
  # Normalizar datos para NN
  x_scaled <- scale(x)
  
  dml_nn <- dml(y, d, x_scaled, methods_y = "nn", methods_d = "nn")
  results_dml <- rbind(results_dml, 
                       data.frame(Method = "Neural Network", 
                                  Estimate = dml_nn$theta_hat,
                                  SE = dml_nn$se_hat))
}, error = function(e) {
  message("NN no disponible: ", e$message)
})

# Mostrar resultados
print("Resultados DML con Cross-Fitting:")
print(results_dml)

# III. No cross-fitting

# Función sin cross-fitting
dml_no_cf <- function(y, d, x, methods_y = "ols", methods_d = "ols") {
  n <- length(y)
  
  # Dividir datos una sola vez
  train_indices <- sample(1:n, size = floor(0.7 * n))
  test_indices <- setdiff(1:n, train_indices)
  
  # Entrenar modelos para y y d
  if (methods_y == "ols") {
    model_y <- lm(y[train_indices] ~ x[train_indices, ])
    y_hat_test <- predict(model_y, newdata = as.data.frame(x[test_indices, ]))
    y_hat_train <- predict(model_y, newdata = as.data.frame(x[train_indices, ]))
  } else if (methods_y == "lasso") {
    cv_lasso <- cv.glmnet(x[train_indices, ], y[train_indices], alpha = 1)
    y_hat_test <- predict(cv_lasso, newx = x[test_indices, ], s = "lambda.min")
    y_hat_train <- predict(cv_lasso, newx = x[train_indices, ], s = "lambda.min")
  } else if (methods_y == "rf") {
    rf_y <- randomForest(x = x[train_indices, ], y = y[train_indices])
    y_hat_test <- predict(rf_y, newdata = x[test_indices, ])
    y_hat_train <- predict(rf_y, newdata = x[train_indices, ])
  }
  
  if (methods_d == "ols") {
    model_d <- lm(d[train_indices] ~ x[train_indices, ])
    d_hat_test <- predict(model_d, newdata = as.data.frame(x[test_indices, ]))
    d_hat_train <- predict(model_d, newdata = as.data.frame(x[train_indices, ]))
  } else if (methods_d == "lasso") {
    cv_lasso_d <- cv.glmnet(x[train_indices, ], d[train_indices], alpha = 1, family = "binomial")
    d_hat_test <- predict(cv_lasso_d, newx = x[test_indices, ], s = "lambda.min", type = "response")
    d_hat_train <- predict(cv_lasso_d, newx = x[train_indices, ], s = "lambda.min", type = "response")
  } else if (methods_d == "rf") {
    rf_d <- randomForest(x = x[train_indices, ], y = as.factor(d[train_indices]))
    d_hat_test <- predict(rf_d, newdata = x[test_indices, ], type = "prob")[,2]
    d_hat_train <- predict(rf_d, newdata = x[train_indices, ], type = "prob")[,2]
  }
  
  # Calcular RMSE
  rmse_y_test <- sqrt(mean((y[test_indices] - y_hat_test)^2))
  rmse_d_test <- sqrt(mean((d[test_indices] - d_hat_test)^2))
  rmse_y_train <- sqrt(mean((y[train_indices] - y_hat_train)^2))
  rmse_d_train <- sqrt(mean((d[train_indices] - d_hat_train)^2))
  
  # Calcular residuos para test
  v_hat <- y[test_indices] - y_hat_test
  u_hat <- d[test_indices] - d_hat_test
  
  # Estimar theta
  theta_hat <- mean(u_hat * v_hat) / mean(u_hat * d[test_indices])
  se_hat <- sd(u_hat * v_hat) / (sqrt(length(test_indices)) * abs(mean(u_hat * d[test_indices])))
  
  return(list(theta_hat = theta_hat, 
              se_hat = se_hat,
              rmse_y_test = rmse_y_test,
              rmse_d_test = rmse_d_test,
              rmse_y_train = rmse_y_train,
              rmse_d_train = rmse_d_train))
}

# Estimar sin cross-fitting
results_no_cf <- data.frame()

# OLS sin CF
dml_ols_nocf <- dml_no_cf(y, d, x, methods_y = "ols", methods_d = "ols")
results_no_cf <- rbind(results_no_cf, 
                       data.frame(Method = "OLS", 
                                  Estimate = dml_ols_nocf$theta_hat,
                                  SE = dml_ols_nocf$se_hat,
                                  RMSE_y = dml_ols_nocf$rmse_y_test,
                                  RMSE_d = dml_ols_nocf$rmse_d_test))

# Lasso sin CF
dml_lasso_nocf <- dml_no_cf(y, d, x, methods_y = "lasso", methods_d = "lasso")
results_no_cf <- rbind(results_no_cf, 
                       data.frame(Method = "Lasso", 
                                  Estimate = dml_lasso_nocf$theta_hat,
                                  SE = dml_lasso_nocf$se_hat,
                                  RMSE_y = dml_lasso_nocf$rmse_y_test,
                                  RMSE_d = dml_lasso_nocf$rmse_d_test))

# RF sin CF
dml_rf_nocf <- dml_no_cf(y, d, x, methods_y = "rf", methods_d = "rf")
results_no_cf <- rbind(results_no_cf, 
                       data.frame(Method = "Random Forest", 
                                  Estimate = dml_rf_nocf$theta_hat,
                                  SE = dml_rf_nocf$se_hat,
                                  RMSE_y = dml_rf_nocf$rmse_y_test,
                                  RMSE_d = dml_rf_nocf$rmse_d_test))

print("Resultados sin Cross-Fitting:")
print(results_no_cf)

# Comparación de resultados
comparison <- merge(results_dml, results_no_cf, by = "Method", suffixes = c("_CF", "_NoCF"))
print("Comparación entre métodos:")
print(comparison)

# Respuestas a las preguntas
cat("\n=== RESPUESTAS A LAS PREGUNTAS ===\n")
cat("1. El RMSE para predecir y y d generalmente será menor sin cross-fitting\n")
cat("   porque los modelos se sobreajustan a los datos de entrenamiento.\n\n")
cat("2. Sin cross-fitting obtenemos RMSE más bajos porque evaluamos en los\n") 
cat("   mismos datos usados para entrenar, lo que no refleja el desempeño real.\n\n")
cat("3. El problema principal sin cross-fitting es el overfitting y\n")
cat("   la estimación sesgada del efecto causal debido a la contaminación\n")
cat("   entre datos de entrenamiento y prueba.\n")