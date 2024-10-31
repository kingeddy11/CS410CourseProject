yelp_rest = read.csv("yelp_restaurants_Phila.csv")
View(yelp_rest)
yelp_rest[, c("BusinessAcceptsCreditCards", "RestaurantsDelivery", 
              "RestaurantsTakeOut", "BusinessParking", "OutdoorSeating", "WiFi",
              "BikeParking", "Caters", "HasTV", "X")] = list(NULL)
View(yelp_rest)
str(yelp_rest)
library(jsonlite)
library(stringr)

clean_col <- function(x) {
  # Remove the 'u' prefix and single quotes
  x <- gsub("u'", "", x)      # Remove 'u' prefix
  x <- gsub("'", "", x)       # Remove single quotes
  x <- gsub(" ", "", x)       # Optionally, remove any spaces
  
  return(trimws(x))           # Return cleaned string without leading/trailing whitespace
}

# Apply the function to the NoiseLevel, Alcohol, RestaurantsAttire and Ambience column
yelp_rest$NoiseLevel <- sapply(yelp_rest$NoiseLevel, clean_col)
yelp_rest$Alcohol <- sapply(yelp_rest$Alcohol, clean_col)
yelp_rest$RestaurantsAttire <- sapply(yelp_rest$RestaurantsAttire, clean_col)
yelp_rest$Ambience2 <- lapply(yelp_rest$Ambience, clean_col)
yelp_rest$GoodForMeal = sapply(yelp_rest$GoodForMeal, clean_col)

convert_to_dict2 <- function(column) {
  # Convert cleaned strings to actual lists
  dict_list <- lapply(column, function(x) {
    # Convert the cleaned string to a named list
    pairs <- strsplit(x, ",")[[1]]  # Split into key-value pairs
    key_value_pairs <- lapply(pairs, function(pair) {
      parts <- strsplit(trimws(pair), ":")[[1]]  # Split each pair into key and value
      key <- trimws(parts[1])  # Remove single quotes from the key
      value <- trimws(parts[2])  # Remove whitespace from the value
      # Convert values to logical or NULL if applicable
      if (tolower(value) == "True") {
        value <- TRUE
      } else if (tolower(value) == "False") {
        value <- FALSE
      } else if (tolower(value) == "None") {
        value <- NULL
      }
      setNames(list(value), key)  # Create a named list
    })
    # Combine all key-value pairs into a single list
    do.call(c, key_value_pairs)
  })
  
  return(dict_list)
}

#Convert Ambience to dict
yelp_rest$Ambience2 <- sapply(yelp_rest$Ambience2, convert_to_dict2)
yelp_rest$GoodForMeal2 = sapply(yelp_rest$GoodForMeal, convert_to_dict2)

#clean dict
clean_dict <- function(named_list) {
  # Clean the names by removing curly braces
  cleaned_names <- gsub("[{}]", "", names(named_list))
  
  # Clean the values by removing any trailing curly braces
  cleaned_values <- sapply(named_list, function(value) gsub("[{}]$", "", value))
  
  # Assign the cleaned names and values back to the list
  cleaned_list <- setNames(as.list(cleaned_values), cleaned_names)
  return(cleaned_list)
}

#Clean Ambience2, GoodForMeal
yelp_rest$Ambience2 <- sapply(yelp_rest$Ambience2, clean_dict)
yelp_rest$GoodForMeal2 = sapply(yelp_rest$GoodForMeal2, clean_dict)


#Check for dictionary 
are_elements_named_lists <- all(sapply(yelp_rest$Ambience2, function(x) {
  is.list(x) && !is.null(names(x))
}))

#names(yelp_rest$Ambience2[[1]])

check_key_true <- function(entry, key) {
  # Ensure the entry is a named list and the key exists in the entry
  if (is.list(entry) && !is.null(entry[[key]])) {
    return(entry[[key]] == "True")
  } else {
    return(FALSE)  # Return FALSE if the entry is not a list or the key is missing
  }
}

# Test the function on the first entry with the key "classy"
result <- check_key_true(yelp_rest[["Ambience2"]][[1]], "classy")

find_true_values <- function(data_frame, column) {
  # Define the possible names
  names <- c("dessert", "latenight", "lunch", "dinner", "brunch", "breakfast")
  
  # Initialize a list to store results for each row
  result <- vector("list", nrow(data_frame))
  
  # Iterate over each row in the data frame
  for (i in 1:nrow(data_frame)) {
    # Initialize a character vector to hold the "true" names for the current row
    true_values <- character()
    
    # Access the dictionary for the current row
    current_row <- data_frame[[column]][[i]]
    
    # Check each name in 'names' and convert values if necessary
    for (name in names) {
      # Check if the key exists in the current row
      if (!is.null(current_row[[name]])) {
        # Convert string representations to logical values
        value <- current_row[[name]]
        if (is.character(value)) {
          if (tolower(value) == "true") {
            value <- TRUE
          } else if (tolower(value) == "false" || tolower(value) == "none") {
            value <- FALSE
          }
        }
        
        # If the value is TRUE, add the name to true_values
        if (isTRUE(value)) {
          true_values <- c(true_values, name)
        }
      }
    }
    
    # Store the true values for the current row in the result list
    result[[i]] <- true_values
  }
  
  return(result)
}


# Apply the function to the ambience column
yelp_rest$Ambience3 <- sapply(1:nrow(yelp_rest), function(i) {
  find_true_values(data_frame = yelp_rest[i, , drop = FALSE], column = "Ambience2")})
yelp_rest$GoodForMeal2 <- sapply(1:nrow(yelp_rest), function(i) {
  find_true_values(data_frame = yelp_rest[i, , drop = FALSE], column = "GoodForMeal2")})

#Convert lists to char
yelp_rest$GoodForMeal2 = sapply(yelp_rest$GoodForMeal2, function(x) paste(unlist(x), collapse = ", "))

#Save the final File
write.csv(yelp_rest, "yelp_restaurants_cleaned2.csv")

# Convert list columns to JSON strings(if needed)
yelp_rest$Ambience3 <- sapply(yelp_rest$Ambience3, toJSON, auto_unbox = TRUE)
yelp_rest$GoodForMeal2 = sapply(yelp_rest$GoodForMeal2, toJSON, auto_unbox = TRUE)

# Convert JSON strings back to lists
yelp_rest$Ambience3 <- lapply(yelp_rest$Ambience3, fromJSON)
yelp_rest$GoodForMeal2 <- lapply(yelp_rest$GoodForMeal2, fromJSON)

