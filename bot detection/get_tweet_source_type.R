require(rlist)
total_list = list()
for (file in list.files("C:/Users/Pierre/Documents/DATACRAFT/Atelier données Twitter/batch_extraction_status_id/")){ 
  load(paste0("C:/Users/Pierre/Documents/DATACRAFT/Atelier données Twitter/batch_extraction_status_id/", file))
  print(length(output_list))
  total_list = list.append(total_list, as.data.frame(output_list[[1]]))
}
status_id_extraction_output <- list.rbind(total_list)
save(status_id_extraction_output, 
     file = "C:/Users/Pierre/Documents/DATACRAFT/Atelier données Twitter/DONNEES/datacraft_status_id_extraction_output.RData")

# Get app source names : ###
source_list = as.data.frame(table(status_id_extraction_output$source))
# writexl::write_xlsx(source_list, 
#                     path = "C:/Users/Pierre/Documents/DATACRAFT/2104-KapCode-Twitter/bot detection/datacraft_tweet_source_2021_10_19.xlsx")


# Get annotated types and merge them to tweets ids  : 
source_types = readxl::read_xlsx( path = "C:/Users/Pierre/Documents/DATACRAFT/2104-KapCode-Twitter/bot detection/datacraft_tweet_source_2021_10_19.xlsx")
source_types = merge(as.data.frame(source_types), 
          status_id_extraction_output[,c("status_id","source")],
          by.x = "tweet_source", by.y = "source")
head(source_types)
source_types <- source_types[,c("status_id","tweet_source","source_type","is_bot")]

sort(table(source_types$source_type))
writexl::write_xlsx(source_types, 
                    path = "C:/Users/Pierre/Documents/DATACRAFT/2104-KapCode-Twitter/bot detection/datacraft_tweet_source_per_status_id_2021_10_19.xlsx")
