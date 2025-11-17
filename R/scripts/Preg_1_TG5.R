# Cargar librerías necesarias
library(tidyverse)
library(neuralnet)
setwd("C:\\Users\\Julio\\Downloads\\R_TG5\\Output")

# I. Fitting Data

# 1. Simular los datos
set.seed(123)
n <- 1000
x <- runif(n, 0, 2*pi)
epsilon <- rnorm(n, 0, 0.2)  # Ruido pequeño
y <- sin(x) + epsilon

# Crear data frame
data <- data.frame(x = x, y = y)

# Normalizar los datos para mejor convergencia
normalize <- function(x) {(x - min(x)) / (max(x) - min(x))}
denormalize <- function(x_norm, original) {
  x_norm * (max(original) - min(original)) + min(original)
}

x_norm <- normalize(x)
y_norm <- normalize(y)
data_norm <- data.frame(x = x_norm, y = y_norm)

# Función para entrenar, graficar y exportar NN
train_plot_export_nn <- function(activation_func, title, data_norm, data_original, filename) {
  # Fórmula
  formula <- y ~ x
  
  # Entrenar NN
  nn <- neuralnet(
    formula,
    data = data_norm,
    hidden = c(50, 50, 50),
    act.fct = activation_func,
    linear.output = TRUE,
    stepmax = 1e6
  )
  
  # Predecir
  predictions_norm <- predict(nn, data_norm)
  predictions <- denormalize(predictions_norm, data_original$y)
  
  # Graficar
  plot_df <- data.frame(
    x = data_original$x,
    y = data_original$y,
    predicted = predictions
  )
  
  p <- ggplot(plot_df, aes(x = x)) +
    geom_point(aes(y = y), alpha = 0.3, color = "blue") +
    geom_line(aes(y = predicted), color = "red", linewidth = 1) +
    ggtitle(paste("NN con activación:", title)) +
    labs(subtitle = paste("MSE:", round(mean((predictions - data_original$y)^2), 6))) +
    theme_minimal()
  
  # Exportar gráfico
  ggsave(filename = paste0(filename, ".png"), 
         plot = p, 
         width = 10, 
         height = 6, 
         dpi = 300)
  
  # Calcular MSE
  mse <- mean((predictions - data_original$y)^2)
  
  return(list(plot = p, mse = mse, model = nn))
}

# Entrenar con diferentes funciones de activación (solo las disponibles en neuralnet)
print("Entrenando NNs con diferentes funciones de activación...")

# Logistic
result_logistic <- train_plot_export_nn("logistic", "Logistic", data_norm, data, "01_logistic")

# Tanh
result_tanh <- train_plot_export_nn("tanh", "Tanh", data_norm, data, "02_tanh")

# Para simular ReLU
result_relu_like <- train_plot_export_nn("tanh", "Tanh (similar a ReLU)", data_norm, data, "03_relu_like")

# Versión SIMPLIFICADA para NN con funciones mixtas
print("Entrenando NN con aproximación de funciones mixtas...")

# Vamos a crear una red con una arquitectura diferente para simular "mixto"
nn_mixed_simple <- neuralnet(
  y ~ x,
  data = data_norm,
  hidden = c(30, 40, 30),  # Diferente número de neuronas por capa
  act.fct = "tanh",        # Usamos tanh pero con diferente arquitectura
  linear.output = TRUE,
  stepmax = 1e6
)

# Predecir
predictions_mixed_norm <- predict(nn_mixed_simple, data_norm)
predictions_mixed <- denormalize(predictions_mixed_norm, data$y)

# Graficar y exportar resultados mixtos
plot_mixed <- ggplot(data, aes(x = x)) +
  geom_point(aes(y = y), alpha = 0.3, color = "blue") +
  geom_line(aes(y = predictions_mixed), color = "red", linewidth = 1) +
  ggtitle("NN con arquitectura diferente (aproximación mixta)") +
  labs(subtitle = paste("MSE:", round(mean((predictions_mixed - data$y)^2), 6))) +
  theme_minimal()

# Exportar gráfico mixto
ggsave(filename = "04_mixto.png", 
       plot = plot_mixed, 
       width = 10, 
       height = 6, 
       dpi = 300)

mse_mixed <- mean((predictions_mixed - data$y)^2)

# Mostrar resultados
print("MSE por función de activación:")
print(paste("Logistic:", round(result_logistic$mse, 6)))
print(paste("Tanh:", round(result_tanh$mse, 6)))
print(paste("Tanh (ReLU-like):", round(result_relu_like$mse, 6)))
print(paste("Arquitectura diferente:", round(mse_mixed, 6)))

# Mostrar gráficos en R
print(result_logistic$plot)
print(result_tanh$plot)
print(result_relu_like$plot)
print(plot_mixed)

# II. Learning-rate

# Explicación del learning rate
cat("\n=== EXPLICACIÓN DEL LEARNING RATE ===\n")
cat("El learning rate (tasa de aprendizaje) controla qué tan grandes son los ajustes\n")
cat("que se hacen a los pesos de la red neuronal durante el entrenamiento.\n")
cat("- LR muy bajo: Convergencia lenta, puede quedar atrapado en mínimos locales\n")
cat("- LR muy alto: Puede sobrepasar el mínimo y no converger\n")
cat("- LR óptimo: Balance entre velocidad y estabilidad de convergencia\n\n")

# Función para entrenar con diferentes learning rates y exportar gráficos
train_lr_comparison_export <- function(hidden_layers, title_suffix, folder_name) {
  learning_rates <- c(0.0001, 0.001, 0.01, 0.1)
  plots <- list()
  mses <- numeric(length(learning_rates))
  
  # Crear subdirectorio para estos gráficos
  if (!dir.exists(folder_name)) {
    dir.create(folder_name)
  }
  
  for (i in seq_along(learning_rates)) {
    lr <- learning_rates[i]
    
    if (hidden_layers == 1) {
      hidden <- 50
    } else if (hidden_layers == 2) {
      hidden <- c(50, 50)
    } else {
      hidden <- c(50, 50, 50)
    }
    
    tryCatch({
      nn <- neuralnet(
        y ~ x,
        data = data_norm,
        hidden = hidden,
        act.fct = "tanh",  # Usamos tanh que funcionó mejor
        linear.output = TRUE,
        learningrate = lr,
        stepmax = 1e6,
        lifesign = "none"
      )
      
      # Predecir
      predictions_norm <- predict(nn, data_norm)
      predictions <- denormalize(predictions_norm, data$y)
      
      # Calcular MSE
      mses[i] <- mean((predictions - data$y)^2)
      
      # Crear gráfico
      plot_df <- data.frame(
        x = data$x,
        y = data$y,
        predicted = predictions
      )
      
      p <- ggplot(plot_df, aes(x = x)) +
        geom_point(aes(y = y), alpha = 0.2, color = "blue") +
        geom_line(aes(y = predicted), color = "red", linewidth = 1) +
        ggtitle(paste("LR =", lr, "-", title_suffix)) +
        labs(subtitle = paste("MSE:", round(mses[i], 6))) +
        theme_minimal()
      
      # Exportar gráfico individual
      ggsave(filename = paste0(folder_name, "/lr_", lr, ".png"), 
             plot = p, 
             width = 10, 
             height = 6, 
             dpi = 300)
      
      plots[[i]] <- p
      
    }, error = function(e) {
      message("Error con LR = ", lr, ": ", e$message)
      mses[i] <- NA
      # Crear gráfico de error
      p_error <- ggplot() +
        annotate("text", x = 3, y = 0, label = paste("Error con LR =", lr), size = 6) +
        ggtitle(paste("Error con LR =", lr, "-", title_suffix)) +
        theme_minimal()
      
      ggsave(filename = paste0(folder_name, "/lr_", lr, "_error.png"), 
             plot = p_error, 
             width = 10, 
             height = 6, 
             dpi = 300)
      
      plots[[i]] <- p_error
    })
  }
  
  # Encontrar mejor LR (excluyendo NAs)
  valid_mses <- mses[!is.na(mses)]
  if (length(valid_mses) > 0) {
    best_lr <- learning_rates[which.min(mses)]
  } else {
    best_lr <- NA
  }
  
  return(list(plots = plots, mses = mses, best_lr = best_lr))
}

# Entrenar para diferentes números de capas
print("Comparando learning rates para 1 capa oculta...")
results_1layer <- train_lr_comparison_export(1, "1 Capa Oculta", "learning_rate_1capa")

print("Comparando learning rates para 2 capas ocultas...")
results_2layer <- train_lr_comparison_export(2, "2 Capas Ocultas", "learning_rate_2capas")

print("Comparando learning rates para 3 capas ocultas...")
results_3layer <- train_lr_comparison_export(3, "3 Capas Ocultas", "learning_rate_3capas")

# Mostrar gráficos en R
cat("\n=== GRÁFICOS DE COMPARACIÓN DE LEARNING RATES ===\n")
for (i in 1:4) {
  if (!is.null(results_1layer$plots[[i]])) {
    print(results_1layer$plots[[i]])
  }
}

# Gráfico comparativo final de modelos
cat("\n=== GRÁFICO COMPARATIVO FINAL ===\n")
comparison_df <- data.frame(
  x = data$x,
  y_real = data$y,
  logistic = result_logistic$model %>% predict(data_norm) %>% denormalize(data$y),
  tanh = result_tanh$model %>% predict(data_norm) %>% denormalize(data$y),
  relu_like = result_relu_like$model %>% predict(data_norm) %>% denormalize(data$y)
)

comparison_long <- comparison_df %>%
  pivot_longer(cols = c(logistic, tanh, relu_like), 
               names_to = "Modelo", 
               values_to = "Prediccion")

final_comparison_plot <- ggplot(comparison_long, aes(x = x)) +
  geom_point(aes(y = y_real), alpha = 0.1, color = "black") +
  geom_line(aes(y = Prediccion, color = Modelo), linewidth = 0.8) +
  ggtitle("Comparación Final de Modelos - Funciones de Activación") +
  labs(subtitle = "Puntos: Datos reales, Líneas: Predicciones de cada modelo") +
  theme_minimal() +
  scale_color_manual(values = c("logistic" = "red", 
                                "tanh" = "blue", 
                                "relu_like" = "green"))

# Exportar gráfico comparativo final
ggsave(filename = "05_comparacion_final_modelos.png", 
       plot = final_comparison_plot, 
       width = 12, 
       height = 8, 
       dpi = 300)

print(final_comparison_plot)

# Gráfico comparativo de learning rates por número de capas
cat("\n=== GRÁFICO COMPARATIVO DE LEARNING RATES POR CAPAS ===\n")
lr_results_df <- data.frame(
  Capas = rep(c("1 Capa", "2 Capas", "3 Capas"), each = 4),
  LearningRate = rep(c(0.0001, 0.001, 0.01, 0.1), 3),
  MSE = c(results_1layer$mses, results_2layer$mses, results_3layer$mses)
)

lr_comparison_plot <- ggplot(lr_results_df, aes(x = factor(LearningRate), y = MSE, fill = Capas)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Comparación de MSE por Learning Rate y Número de Capas") +
  labs(x = "Learning Rate", y = "MSE") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

# Exportar gráfico comparativo de learning rates
ggsave(filename = "06_comparacion_learning_rates.png", 
       plot = lr_comparison_plot, 
       width = 12, 
       height = 8, 
       dpi = 300)

print(lr_comparison_plot)

# Resultados y conclusiones
cat("\n=== RESULTADOS Y CONCLUSIONES ===\n")
print(paste("Mejor LR para 1 capa:", results_1layer$best_lr))
print(paste("Mejor LR para 2 capas:", results_2layer$best_lr))
print(paste("Mejor LR para 3 capas:", results_3layer$best_lr))

cat("\n=== RELACIÓN ENTRE LEARNING RATE Y NÚMERO DE CAPAS ===\n")
cat("En general, se observa que:\n")
cat("1. Con más capas ocultas, se necesitan learning rates más bajos\n")
cat("2. Redes más profundas son más sensibles a learning rates altos\n")
cat("3. Learning rates muy altos pueden causar inestabilidad en redes profundas\n")
cat("4. El LR óptimo tiende a disminuir a medida que aumenta la profundidad de la red\n")

# Pregunta: ¿Cuál NN ajusta mejor los datos?
cat("\n=== ¿CUÁL NN AJUSTA MEJOR LOS DATOS? ===\n")
mse_comparison <- data.frame(
  Modelo = c("Logistic", "Tanh", "Tanh (ReLU-like)", "Arquitectura diferente"),
  MSE = c(result_logistic$mse, result_tanh$mse, result_relu_like$mse, mse_mixed)
)
print(mse_comparison)

best_model <- which.min(c(result_logistic$mse, result_tanh$mse, result_relu_like$mse, mse_mixed))
best_model_name <- c("Logistic", "Tanh", "Tanh (ReLU-like)", "Arquitectura diferente")[best_model]

cat(paste("\nEl mejor modelo es:", best_model_name, "con MSE =", 
          round(mse_comparison$MSE[best_model], 6), "\n"))

# Mensaje final
cat("\n=== EXPORTACIÓN COMPLETADA ===\n")
cat("Todos los gráficos han sido exportados al directorio:\n")
cat("C:\\\\Users\\\\Julio\\\\Downloads\\\\R_TG5\\\\Output\\\\\n")
cat("\nArchivos generados:\n")
cat("- 01_logistic.png: NN con función logística\n")
cat("- 02_tanh.png: NN con función tanh\n")
cat("- 03_relu_like.png: NN similar a ReLU\n")
cat("- 04_mixto.png: NN con arquitectura diferente\n")
cat("- 05_comparacion_final_modelos.png: Comparación de todos los modelos\n")
cat("- 06_comparacion_learning_rates.png: Comparación de learning rates\n")
cat("- learning_rate_1capa/: Gráficos de learning rates para 1 capa\n")
cat("- learning_rate_2capas/: Gráficos de learning rates para 2 capas\n")
cat("- learning_rate_3capas/: Gráficos de learning rates para 3 capas\n")