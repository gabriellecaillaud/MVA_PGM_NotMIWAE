# Required packages
library(softImpute)
library(missMDA)
library(missForest)
library(mice)
library(data.table)  # For reading the CSV
library(Matrix)      # For matrix operations
library(torch)
library(imager)
library(colorspace)
library(recolorize)

create_mnar_missing <- function(data, hue, sat, val, means, stds) {
  # Check if data is a matrix
  if (!is.matrix(data)) {
    stop("Input data must be a matrix")
  }
  
  # Get the number of rows and columns in the matrix
  n_rows <- nrow(data)
  n_cols <- ncol(data)
  n = as.integer(n_cols/3)

  rescaled_data <- sweep(sweep(data, 2, stds, "*"), 2, means, "+")
  
  # Loop over the target columns (1, 4, 7, ...)
  for (i in seq(1, n)) {
    # Replace values between hue_min and hue_max with NA
    cnd = rescaled_data[, i] >= hue[1] & rescaled_data[, i] <= hue[2] & rescaled_data[, i+1024] >= sat[1] & rescaled_data[, i+1024] <= sat[2] & rescaled_data[, i+2048] >= sat[1] & rescaled_data[, i+2048] <= sat[2]
    data[cnd, i] <- NA
    data[cnd, i+1024] <- NA
    data[cnd, i+2048] <- NA
  }
  
  return(data)
}


# Function to impute missing values using different methods
impute_missing_data <- function(data, method = "softimpute", rank = 2, mice_method="norm") {
  # Input validation
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("Data must be a matrix or data frame")
  }
  data <- as.matrix(data)
  
  # Choose imputation method
  result <- switch(method,
    "mean" = {
      # Simple mean imputation
      for(j in 1:ncol(data)) {
        data[is.na(data[,j]), j] <- mean(data[,j], na.rm = TRUE)
      }
      data
    },
    "softimpute" = {
      # SVD-based soft imputation
      fit <- softImpute(data, rank = rank, lambda = 0, maxit = 1000, type = "svd")
      if (length(fit$d) == 1) {
        imputed <- (fit$u * fit$d) %*% t(fit$v)
      } else {
        imputed <- (fit$u %*% diag(fit$d)) %*% t(fit$v)
      }
      imputed
    },
    "pca" = {
      # Iterative PCA imputation
      imputed <- imputePCA(data, ncp = rank, maxiter = 1000)$completeObs
      imputed
    },
    "missForest" = {
      # Random Forest imputation
      imputed <- missForest(data, ntree = 100)$ximp
      imputed
    },
    "mice" = {
      # MICE
      imputed <- complete(mice(data, m=5, maxit=1000, seed=123, method=mice_method, printFlag = FALSE), 1)
      imputed
    },
    stop("Unknown method. Choose 'mean', 'softimpute', 'pca', 'rf' or 'mice'")
  )
  
  return(result)
}

# Function to calculate MSE
calculate_rmse <- function(true_data, imputed_data, missing_mask) {
  # Calculate MSE only for imputed values
  result <- try({
    rmse <- sqrt(mean((true_data[missing_mask] - imputed_data[missing_mask])^2))
  }, silent = TRUE)

  # Check if an error occurred
  if (inherits(result, "try-error")) {
    print("An error occurred!")
    print(result)
    print(imputed_data)
    return(-1)
  } else {
    return(rmse)
  }

}


# Function to convert an image from RGB to HSI
rgb_to_hsi_image <- function(image_row) {
  # Reshape the row (1x3072) into a 32x32x3 array
  
  # Convert each pixel to HSI
  hsi_image <- array(dim = c(3072))
  for (i in 1:32) {
    for (j in 1:32) {
      hsi <- as(RGB(image_row[(i-1)*32 + j], image_row[1024+ (i-1)*32 + j], image_row[2048 + (i-1)*32 + j]), "HSV")
      hsi_image[(i-1)*32 + j] <- pmax(0, pmin(1, as.numeric(hsi@coords[1])/360))  # Hue
      hsi_image[(i-1)*32 + j + 1024] <- pmax(0, pmin(1, as.numeric(hsi@coords[2])))  # Saturation
      hsi_image[(i-1)*32 + j + 2048] <- pmax(0, pmin(1, as.numeric(hsi@coords[3])))  # Intensity
    }
  }
  return(hsi_image)
}

hsi_to_rgb_image <- function(image_row) {
  # Reshape the row (1x3072) into a 32x32x3 array
  # image_array <- reshape_single_image(image_row)
  
  # Convert each pixel to HSI
  rgb_image <- array(dim = c(3072))
  for (i in 1:32) {
    for (j in 1:32) {
      rgb <- as(HSV(image_row[(i-1)*32 + j]*360, image_row[1024+ (i-1)*32 + j], image_row[2048 + (i-1)*32 + j]), "RGB")
      rgb_image[(i-1)*32 + j] <- pmax(0, pmin(1, as.numeric(rgb@coords[1])))  # Hue
      rgb_image[(i-1)*32 + j + 1024] <- pmax(0, pmin(1, as.numeric(rgb@coords[2])))  # Saturation
      rgb_image[(i-1)*32 + j + 2048] <- pmax(0, pmin(1, as.numeric(rgb@coords[3])))  # Intensity
    }
  }
  return(rgb_image)
}

bool_to_color <- function(image_row, color=c(1, 0, 0)) {
  # Reshape the row (1x3072) into a 32x32x3 array
  # image_array <- reshape_single_image(image_row)
  
  # Convert each pixel to HSI
  rgb_image <- array(0, dim = c(3072))
  for (i in 1:32) {
    for (j in 1:32) {
      if (image_row[(i-1)*32 + j]){
        rgb_image[(i-1)*32 + j] = color[1]
        rgb_image[(i-1)*32 + j + 1024] = color[2]
        rgb_image[(i-1)*32 + j + 2048] = color[3]
      }
    }
  }
  return(rgb_image)
}

# load_cifar10_batch <- function(filename) {
#   con <- file(filename, "rb")
#   n <- 10000  # Each batch has 10,000 images
#   # Read labels and images
#   labels <- readBin(con, "long", n = n, size = 1, signed = TRUE)
#   data <- matrix(readBin(con, "long", n = n * 32 * 32 * 3, size = 1, signed = FALSE), ncol = 32 * 32 * 3, byrow = TRUE)
#   close(con)
#   list(data = data, labels = labels)
# }

load_cifar10_batch <- function(filename) {
  con <- file(filename, "rb")
  n <- 1000  # Each batch has 10,000 images
  
  # Read all bytes at once (1 byte per label + 3072 bytes per image)
  raw_data <- readBin(con, "raw", n = n * (1 + 3072), size = 1)
  close(con)
  
  # Extract labels (every 3073rd byte starting from the first)
  label_indices <- seq(1, length(raw_data), 3073)
  labels <- as.integer(raw_data[label_indices])
  
  # Extract image data (all bytes except labels)
  image_indices <- setdiff(1:length(raw_data), label_indices)
  data <- matrix(as.integer(raw_data[image_indices]), nrow = n, ncol = 3072, byrow = TRUE)
  
  return(list(data = data, labels = labels))
}


displayImage <- function(image_array) {
  # Create a blank plot area
  plot(1, type = "n", xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
  
  # Use rasterImage to display the image without antialiasing
  rasterImage(image_array, 0, 0, 1, 1, interpolate = FALSE)
}

save_imputation <- function(means, stds, images, imputed_data, missing_mask, n_col, hue, method, title=""){
  original_imputed_data <- sweep(sweep(imputed_data, 2, stds, "*"), 2, means, "+")

  imputed_images <- lapply(1:nrow(original_imputed_data), function(i) {
    reshape_single_image(hsi_to_rgb_image(original_imputed_data[i, ]))
  })
  true_images <- lapply(1:nrow(images), function(i) {
    reshape_single_image(images[i, ])
  })
  masks <- lapply(1:nrow(missing_mask), function(i) {
    reshape_single_image(missing_mask[i, ])
  })

  for (it in 0:as.integer((n_col-1)/3)){
    local_col = min(3,n_col-3*it)
    L = 3*local_col
    L2 = local_col
    
    png(paste("result", method, title, it+1, ".png", sep="_"), width = 800, height = 800, res = 300)  # Adjust width, height, and resolution as needed
    par(mfrow = c(L2 * 3, L), mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
    color <- as(HSV(mean(hue) * 360, 1, 1), "RGB")
    for (j in 0:(L2-1)) {
      for (i in (1+27*it):(L+27*it)) {
        displayImage(true_images[[i + L * j]])  # Replace with your specific plotting function
      }
      for (i in (1+27*it):(L+27*it)) {
        arr <- true_images[[i + L * j]]
        for (channel in 1:3) {
          arr[,,channel][masks[[i + L * j]][,,channel] == 1] <- color@coords[channel]
        }
        displayImage(arr)  # Replace with your specific plotting function
      }
      for (i in (1+27*it):(L+27*it)) {
        displayImage(imputed_images[[i + L * j]])  # Replace with your specific plotting function
      }
    }
    dev.off()
  }
}

reshape_single_image <- function(flat_array) {
  # Split into RGB channels (1024 each)
  red <- flat_array[1:1024]
  green <- flat_array[1025:2048]
  blue <- flat_array[2049:3072]
  
  # Create 32x32x3 array
  result <- array(0, dim = c(32, 32, 3))
  
  # Reshape each channel
  result[,,1] <- matrix(red, nrow = 32, ncol = 32, byrow = TRUE)
  result[,,2] <- matrix(green, nrow = 32, ncol = 32, byrow = TRUE)
  result[,,3] <- matrix(blue, nrow = 32, ncol = 32, byrow = TRUE)
  
  return(result)
}

# Main function to run the analysis
analyze_imputation_methods <- function(data_path, methods=c("mean"), rank = 10, N=10, hue=c(180/360, 260/360), sat=c(0.3, 1), val=c(0.3, 1), n_col=3, title="") {
  # Read the data
  
  # Load the data
  # url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
  batch <- load_cifar10_batch("./datasets/cifar10/data_batch_1.bin")
  images <- batch$data/255
  # images = images[1:N,]
  # plotImageArray(reshape_single_image(images[10,]))

  hsi_matrix <- do.call(rbind, lapply(1:N, function(i) rgb_to_hsi_image(images[i, ])))
  # Calculate means and standard deviations for scaling
  means <- colMeans(hsi_matrix)
  stds <- apply(hsi_matrix, 2, sd)

  # Scale the HSI data
  scaled_hsi_matrix <- scale(hsi_matrix)
  # To reverse the scaling for visualization:
  original_hsi_matrix <- sweep(sweep(scaled_hsi_matrix, 2, stds, "*"), 2, means, "+")

  data <- scaled_hsi_matrix 
  # ---- Random permutation
  set.seed(123)  # To ensure reproducibility
  N <- nrow(data)
  p <- sample(N)  # Random permutation of row indices
  data <- data[p, ]
  images <- images[p, ]

  colnames(data) <- gsub(" ", "_", colnames(data))

  # Remove any existing missing values
  data <- na.omit(data)

  # Create MNAR missing data
  data_with_missing <- create_mnar_missing(data, hue, sat, val, means, stds)
  # Create missing mask (TRUE where values are missing)
  missing_mask <- is.na(data_with_missing)

  results <- data.frame(
    method = methods,
    mse = NA
  )
  
  # Run each method and calculate MSE
  for(method in methods) {
    cat(sprintf("\nRunning %s imputation...\n", method))
    
    # Impute data
    imputed_data <- impute_missing_data(data_with_missing, method = method, rank = rank, mice_method = "norm")
    
    # Calculate MSE
    mse <- calculate_rmse(data, imputed_data, missing_mask)
    
    # Store results
    results$mse[results$method == method] <- mse
    
    cat(sprintf("MSE for %s: %.4f\n", method, mse))

    save_imputation(means, stds, images, imputed_data, missing_mask, n_col, hue, method, title=title)
  }
  
  return(list(
    results = results,
    images = images
  ))
}

# Run the analysis
hue=c(200/360, 260/360)
sat=c(0.3, 1)
val=c(0.5, 1)
n_col = 18
N=3*(n_col**2)
title="bleu"
methods = c("softimpute")#, "mean", "missForest")#, "mice")
print(paste("Running with methods =", methods, "  - N =", N))
results <- analyze_imputation_methods("datasets/cancer-dataset/Cancer_Data.csv", methods=methods, rank=400, N=N, hue=hue, sat=sat, val=val, n_col=n_col, title=title)
