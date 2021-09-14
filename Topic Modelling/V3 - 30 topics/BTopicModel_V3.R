#### V3 of the topic model #######
require(ggplot2)
require(scales)
require(dplyr)
require(tidytext)
require(tm)
require(stringr)
require(BTM)
library(udpipe)
require(rlist)
## Topic modelling 

### Preprocessing 
dataset_clean_for_TM <- dataset_clean[!dataset_clean$is_rt,]
dataset_clean_for_TM$tweet_formatted_for_TM <- dataset_clean_for_TM$tweet_formatted
dataset_clean_for_TM$tweet_formatted_for_TM <- unname(sapply(dataset_clean_for_TM$tweet_formatted_for_TM, function(x) {gsub(x = x, pattern = "(@.*?\\s{1})", replacement = "")}))
dataset_clean_for_TM$tweet_formatted_for_TM <- sapply(dataset_clean_for_TM$tweet_formatted_for_TM, str_replace_all, pattern = "http.*?\\s|http.*?$", replacement = "")

### Tokenization and stopwords deletion 
clean_stopwords <- function(stopwords){
  stopwords <- iconv(stopwords, from = 'UTF-8', to='ASCII//TRANSLIT')
  stopwords <- str_replace_all(stopwords, "[[:punct:]]", "")     
  return(stopwords)
}
basic_stopwords <- clean_stopwords(stopwords(kind = "fr"))
d <- dataset_clean_for_TM %>%  unnest_tokens(word, tweet_formatted_for_TM) %>% filter(!word %in% basic_stopwords)

###  Add some stopwords 
additional_stopwords <- c("si","ca","va","ni","etc","via","",'vaccin',"vaccination","vaccins","vacciner","bill","gates",
                          "contre","etre","fait","tout","tous","vont","faire","non", "plus", "bien", "19", "covid", "covid19",
                          "oui", "aussi", "donc", "alors", "rien", "coronavirus", "quand", "veut", "dit", "sous", 
                          "deja", "faut", "peut", "apres", "avoir", "encore", "veulent", "quoi", "ceux", "tres", "bon" ,"depuis",
                          "autres", "vaccines", "puce", "puces", "pucages", "pucer", "pucage", "car", "quoi", "dire", "avant", 
                          "jamais", "toujours", "moins", "comment") 
additional_stopwords <- c(additional_stopwords, quanteda::stopwords(language = "en"))
additional_stopwords <- clean_stopwords(additional_stopwords)
d <- d %>% filter(!word %in% additional_stopwords)
d <- d[which(nchar(d$word)>1),] 
d <- d[,c("status_id" ,"word")]
View(table(d$word))


### Modelization 
set.seed(123)
K_ = 30
model  <- BTM(d, k = K_, alpha = 1, beta = 0.01, iter = 100, trace = F)
model

### Extract results 
BTM_terms <- terms(model,top_n = 20)
BTM_terms <- list.cbind(BTM_terms)[seq(1,(K_*2),2)] # Top terms 
BTM_predictions <- data.frame(predict(model, newdata = d)) # Topic repartition per tweet

### Test de SpeedReader 
# require(SpeedReader)
# # d.alt <- d %>% group_by(status_id, word) %>% summarise_all(count)
# d.alt <- d
# colnames(d.alt) <- c("doc_id","term")
# d.alt$freq <- 1
# d.dtm <- document_term_matrix(d.alt, weight = "freq")
# test <- SpeedReader::topic_coherence(top_words = BTM_terms, document_term_matrix = d.dtm, vocabulary = d.dtm@Dimnames[[1]])
# 


# Associate top topic to each tweet : 
association_threshold = .2
number_of_topics_per_message = 1
topics_per_tweets <- cbind(t(apply(BTM_predictions, 1, function(row, number_of_topics_per_message, 
                                                                association_threshold) {
  row_order <- order(row, decreasing = T)
  row_order.values <- row[row_order]
  row_order[which(row_order.values < association_threshold)] <- 0
  row_order.values[which(row_order.values < association_threshold)] <- 0
  output <- c(row_order[1:number_of_topics_per_message], 
              row_order.values[1:number_of_topics_per_message])
  return(output)
}, number_of_topics_per_message, association_threshold)))
topics_per_tweets <- as.data.frame(topics_per_tweets)
colnames(topics_per_tweets) <- c("topic_id","topic_proportion")
topics_per_tweets$status_id <- rownames(topics_per_tweets)
rownames(topics_per_tweets) <- NULL
dataset_clean_with_TM <- merge(dataset_clean_for_TM, topics_per_tweets, by = 'status_id', all.x = T, all.y = F)

# Topics proportions : 
table(dataset_clean_with_TM$topic_id, useNA = "ifany")
topics_proportions <- data.frame("topic_id" = c(0, seq(K_),"NA"), 
                                 "top_terms" = c("No topic",apply(BTM_terms[1:10,], 2, paste, collapse = ", "),"Deleted posts"),
                                 "N_posts" = as.data.frame(table(dataset_clean_with_TM$topic_id, useNA = "always"))[,2],
                                 "P_posts" = as.data.frame(table(dataset_clean_with_TM$topic_id, useNA = "always")/nrow(dataset_clean_with_TM)*100)[,2],
                                 "examples" = c(sapply(0:K_, function(t_){
                                   out <- paste(dataset_clean_with_TM[dataset_clean_with_TM$status_id %in% sample(dataset_clean_with_TM[which(dataset_clean_with_TM$topic_id == t_),"status_id"],5),
                                                                      c("tweet")], collapse = ' \n ')
                                   return(out)
                                 } ), NA)
)

## Rapport user/nb_posts 
topics_proportions$nb_users <- c(sapply(0:K_, function(t_){
  out <- length(unique(dataset_clean_with_TM[dataset_clean_with_TM$topic_id == t_,"pseudo"]))
  return(out)
} ), NA)
topics_proportions$post_per_user <- topics_proportions$N_posts/topics_proportions$nb_users


# Output topic proportions to label them : 
write.csv2(topics_proportions, "topics_proportions.30topics.csv")



temp <- dataset_clean_with_TM[which(grepl(dataset_clean_with_TM$pseudo, pattern = 'Mariette' )),]
# Exemple de tweets par topic, utilisés pour labelliser les topics aussi : 
t_ = 6
for (t_ in 1:K_) print(dataset_clean_with_TM[dataset_clean_with_TM$status_id %in% sample(dataset_clean_with_TM[which(dataset_clean_with_TM$topic_id == t_),"status_id"],5), c("tweet")])



# oUTPUT WITHOUT manual label :
dataset_clean_TM_wo_labels <- dataset_clean_with_TM[,c("status_id","topic_id", "tweet")]

# Output topics results 
write.csv2(dataset_clean_TM_wo_labels, "topics_per_tweet.30topics_no_labels.csv")
writexl::write_xlsx(dataset_clean_TM_wo_labels, path = "topics_per_tweet.30topics_no_labels.xlsx")
save(model ,
     BTM_terms ,
     BTM_predictions ,
     topics_proportions, file = "BTM_V3_30topics.RData"
)


read.csv2("https://github.com/datacraft-paris/2104P_KapCode_Twitter/blob/main/topics_proportions.30topics.csv")

# Get manual label :
topic_labels <- read.csv2("topics_proportions.30topics.ok.csv", stringsAsFactors = F)
topic_labels <- topic_labels[,c("topic_id", "Label")]
dataset_clean_with_TM <- merge(dataset_clean_with_TM, topic_labels, "topic_id")
dataset_clean_TM <- dataset_clean_with_TM[,c("status_id","topic_id", "Label", "tweet")]

# Output topics results 
write.csv2(dataset_clean_TM, "topics_per_tweet.30topics.csv")
writexl::write_xlsx(dataset_clean_TM, path = "topics_per_tweet.30topics.xlsx")
save(model ,
     BTM_terms ,
     BTM_predictions ,
     topics_proportions, file = "BTM_V3_30topics.RData"
)


