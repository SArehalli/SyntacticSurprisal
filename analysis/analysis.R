library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)
library(ggtext)
library(stringr)
library(lme4)
library(lmerTest)
library(emmeans)

#------
# Load in Surprisal Data
surps_0 <- read.csv("../surps/items_ClassicGP.ambig.csv.m0")
surps_1 <- read.csv("../surps/items_ClassicGP.ambig.csv.m1")
surps_2 <- read.csv("../surps/items_ClassicGP.ambig.csv.m2")
surps_3 <- read.csv("../surps/items_ClassicGP.ambig.csv.m3")

surps_0$model <- "m0"
surps_1$model <- "m1"
surps_2$model <- "m2"
surps_3$model <- "m3"

bound <- rbind(surps_0, surps_1, surps_2, surps_3) 

bound_long <- bound %>% 
  gather("surprisal_type", "surprisal", 
         c("sum_lex_surprisal", "sum_syn_surprisal", "mean_lex_surprisal", "mean_syn_surprisal"))


bound_long$word_pos_ralign <- bound_long$word_pos - bound_long$disambPosition_0idx
# ggplot(subset(bound_long, bound_long$surprisal_type %in% c("sum_lex_surprisal")), 
#        aes(x=word_pos_ralign, y=surprisal, color=ambiguity, linetype=ambiguity)) + 
#   scale_x_continuous(limits=c(-1,2)) + scale_x_continuous(limits=c(-1.3,2.3), breaks=c(0,1,2), labels=c("Disambig", "Spillover 1", "Spillover 2")) +
#   facet_grid(condition~surprisal_type) + stat_summary(geom="line") + stat_summary(geom="errorbar", fun.data=mean_cl_boot, width=0.3) + theme_classic()
# ggsave("plots/sumlex_bywordpos.pdf", height=6, width=8)
# ggplot(subset(bound_long, bound_long$surprisal_type %in% c("sum_syn_surprisal")), 
#        aes(x=word_pos_ralign, y=surprisal, color=ambiguity, linetype=ambiguity)) + 
#   scale_x_continuous(limits=c(-1,2)) + scale_x_continuous(limits=c(-1.3,2.3), breaks=c(0,1,2), labels=c("Disambig", "Spillover 1", "Spillover 2")) +
#   facet_grid(condition~surprisal_type) + stat_summary(geom="line") + stat_summary(geom="errorbar", fun.data=mean_cl_boot, width=0.3) + theme_classic()
# ggsave("plots/sumsyn_bywordpos.pdf", height=6, width=8)


spr <- read.csv("./ClassicGardenPathSet.csv") %>% filter(RT < 3000) %>% filter(RT > 0)
spr$Sentence <- str_replace_all(spr$Sentence, "%2C", ",")
spr$EachWord <- str_replace_all(spr$EachWord, "%2C", ",")

fspr <- read.csv("./Fillers.csv") %>% filter(RT < 3000) %>% filter(RT > 0)
fspr$Sentence <- str_replace_all(fspr$Sentence, "%2C", ",")
fspr$EachWord <- str_replace_all(fspr$EachWord, "%2C", ",")

# ggplot(spr, aes(x=ROI, y=RT, color=AMBIG)) + 
#   scale_x_continuous(limits=c(-1.3,2.3), breaks=c(0,1,2), 
#                      labels=c("Disambig", "Spillover 1", "Spillover 2")) +
#   facet_grid(CONSTRUCTION~.) + stat_summary(geom="line") + 
#   stat_summary(geom="errorbar", fun.data=mean_cl_boot, width=0.3) + theme_classic()
# ggsave("plots/spr_gp.pdf", height=6, width=8)

bound_long_effs <- bound_long %>% pivot_wider(id_cols=c("item", "condition", "word_pos_ralign", "word", "model"),
                                      names_from=c("ambiguity", "surprisal_type"), 
                                      values_from=c("surprisal")) %>%
                          mutate(sum_lex_surp_diff = ambiguous_sum_lex_surprisal - unambiguous_sum_lex_surprisal,
                                 sum_syn_surp_diff = ambiguous_sum_syn_surprisal - unambiguous_sum_syn_surprisal) %>%
                          gather("surprisal_type", "surprisal_diff", c("sum_lex_surp_diff", "sum_syn_surp_diff"))

ggplot(subset(bound_long_effs, (bound_long_effs$word_pos_ralign==0) & 
                           (bound_long_effs$surprisal_type=="sum_lex_surp_diff")), 
       aes(x=condition, y=surprisal_diff, fill=condition)) + 
  stat_summary(geom="bar") + stat_summary(geom="errorbar", fun.data=mean_cl_boot, 
                                          position=position_dodge(width=0.9), width=0.3) + theme_classic()
ggsave("plots/lex_gp.pdf")
ggplot(subset(bound_long_effs, (bound_long_effs$word_pos_ralign==0) & 
                           (bound_long_effs$surprisal_type=="sum_syn_surp_diff")),
       aes(x=condition, y=surprisal_diff, fill=condition)) + 
  stat_summary(geom="bar") + stat_summary(geom="errorbar", fun.data=mean_cl_boot, 
                                        position=position_dodge(width=0.9), width=0.3) + 
  theme_classic() + theme(legend.position = "none") +
  labs(x="Garden Path Type", y = "Garden Path Effect Size")
ggsave("plots/syn_gp.pdf", height=6, width=8)

ggplot(subset(bound_long_effs, (bound_long_effs$word_pos_ralign >= -1) & 
                (bound_long_effs$word_pos_ralign <= 2)),
       aes(x=condition, y=surprisal_diff, fill=as.factor(word_pos_ralign))) + 
  stat_summary(geom="bar", position=position_dodge(width=0.9)) + 
  stat_summary(geom="errorbar", fun.data=mean_cl_boot, 
               position=position_dodge(width=0.9), width=0.3) + 
  scale_x_discrete(labels=c("MVRR_UAMB"="MVRR", "NPS_UAMB"="NPS", "NPZ_UAMB"="NPZ")) +
  scale_fill_viridis_d(labels=c("Pre-disambig", "Disambig", "Spillover 1", "Spillover 2")) +
  theme_classic() + 
  facet_grid(~surprisal_type, labeller = 
             as_labeller(c("sum_lex_surp_diff"="Lexical Surprisal", 
                           "sum_syn_surp_diff"="Syntactic Surprisal"))) +
  labs(x="Garden Path Type", y = "Garden Path Effect (bits)", fill="Position")
ggsave("plots/surp_gps.pdf", height=2.5, width=8)

spr_effs <- spr %>% group_by(ROI, CONSTRUCTION, item, AMBIG) %>%
                    summarize(RT = mean(RT)) %>%
                    pivot_wider(id_cols = c("ROI", "CONSTRUCTION", "item"),
                                names_from = c("AMBIG"), values_from=c("RT")) %>%
                    mutate(rt_diff = Amb-Unamb) 

# ggplot(subset(spr_effs, ROI %in% c(0,1,2)), aes(x=CONSTRUCTION, y=rt_diff, fill=CONSTRUCTION, group=ROI)) + 
#   stat_summary(geom="bar", position=position_dodge2()) + 
#   stat_summary(geom="errorbar", fun.data=mean_cl_boot, 
#                position=position_dodge(width=0.9), width=0.3) + 
#   theme_classic() + theme(legend.position = "none") +
#   labs(x="Garden Path Type", y = "Garden Path Effect Size")
# 
# ggsave("plots/spr_effect.pdf", height=6, width=8)

fsurps_0 <- read.csv("../surps/items_filler.ambig.csv.m0")
fsurps_1 <- read.csv("../surps/items_filler.ambig.csv.m1")
fsurps_2 <- read.csv("../surps/items_filler.ambig.csv.m2")
fsurps_3 <- read.csv("../surps/items_filler.ambig.csv.m3")

fsurps_0$model <- "m0"
fsurps_1$model <- "m1"
fsurps_2$model <- "m2"
fsurps_3$model <- "m3"

fbound <- rbind(fsurps_0, fsurps_1, fsurps_2, fsurps_3)

freqs <- read.csv("./freqs_coca.csv")
fbound$word_clean <- str_replace_all(tolower(fbound$word), "[.,!?:;]", "")
fbound <- merge(fbound, freqs, by.x = "word_clean", by.y="word", all.x=TRUE)

# Process for finding example sentences
# fbound %>% filter((fbound$sum_lex_surp > 12) & (fbound$sum_syn_surp < 2) & (fbound$token != "<UNK>"))
# fbound %>% filter((fbound$sum_syn_surp > 7) & (fbound$token != "<UNK>") & (fbound$model == "m0"))

fcorrplot_df <- fbound
fcorrplot_df$highlight <- 0
fcorrplot_df$highlight <- fcorrplot_df$highlight | (fcorrplot_df$token == "microbes")
fcorrplot_df$highlight <- fcorrplot_df$highlight | ((fcorrplot_df$item == "98") & (fcorrplot_df$word == "trying"))

highlights <- fcorrplot_df %>% filter((fcorrplot_df$highlight == 1) & (fcorrplot_df$model == "m3"))
ggplot(fcorrplot_df %>% filter((fcorrplot_df$sum_lex_surprisal < 50) & (fcorrplot_df$token != "<UNK>")), 
       aes(x=sum_lex_surprisal, y=sum_syn_surprisal, color=highlight)) + 
  facet_grid(~model) + geom_point(size=0.3) + theme_classic() + coord_cartesian(xlim=c(0,25), clip="off") +
  scale_color_manual(values=c("black", "red")) +
  labs(x="Lexical Surprisal", y="Syntactic Surprisal") +
  geom_textbox(data=highlights, aes(label=stringr::str_wrap(stringr::str_replace(Sentence, word, paste0("**", word, "**")), width=40)), 
             size=2.5, hjust=1, vjust=0.5, color="black", alpha=0.9, xlim = c(-Inf, Inf), ylim = c(-Inf, Inf)) + theme(legend.position="none")
ggsave("plots/filler_corrplot.pdf", height=2.5, width=8)

ggplot(fcorrplot_df %>% filter((fcorrplot_df$sum_lex_surprisal < 50) & (fcorrplot_df$token != "<UNK>") & (fcorrplot_df$model == "m3")), 
       aes(x=sum_lex_surprisal, y=sum_syn_surprisal, color=highlight)) + 
  geom_point(size=2) + theme_classic() + coord_cartesian(xlim=c(0,20), clip="off") +
  scale_color_manual(values=c("black", "red")) +
  labs(x="Lexical Surprisal", y="Syntactic Surprisal") +
  geom_textbox(data=highlights, aes(label=stringr::str_wrap(stringr::str_replace(Sentence, word, paste0("**", word, "**")), width=40)), 
               size=2.5, hjust=0.5, vjust=0, color="black", alpha=0.9, xlim = c(-Inf, Inf), ylim = c(-Inf, Inf)) + theme(legend.position="none")
ggsave("plots/filler_corrplot_one.pdf", height=4, width=4)

ggplot(fcorrplot_df %>% filter((fcorrplot_df$sum_lex_surprisal < 50) & (fcorrplot_df$token != "<UNK>")), 
       aes(x=log(count), y=sum_syn_surprisal, color=highlight)) + 
  facet_grid(~model) + geom_point(size=0.3) + theme_classic() + coord_cartesian(clip="off") +
  scale_color_manual(values=c("black", "red")) +
  labs(x="log(count)", y="Syntactic Surprisal") +
  geom_textbox(data=highlights, aes(label=stringr::str_wrap(stringr::str_replace(Sentence, word, paste0("**", word, "**")), width=40)), 
               size=2.5, hjust=1, vjust=0.5, color="black", alpha=0.9, xlim = c(-Inf, Inf), ylim = c(-Inf, Inf)) + theme(legend.position="none")
ggsave("plots/filler_corrplot_freq.pdf", height=2.5, width=8)

ggplot(fcorrplot_df %>% filter((fcorrplot_df$sum_lex_surprisal < 50) & (fcorrplot_df$token != "<UNK>") & (fcorrplot_df$model == "m3")), 
       aes(x=log(count), y=sum_syn_surprisal, color=highlight)) + 
  geom_point(size=2) + theme_classic() + coord_cartesian(clip="off") +
  scale_color_manual(values=c("black", "red")) +
  labs(x="log(count)", y="Syntactic Surprisal") +
  geom_textbox(data=highlights, aes(label=stringr::str_wrap(stringr::str_replace(Sentence, word, paste0("**", word, "**")), width=40)), 
               size=2.5, hjust=0.5, vjust=0, color="black", alpha=0.9, xlim = c(-Inf, Inf), ylim = c(-Inf, Inf)) + theme(legend.position="none")
ggsave("plots/filler_corrplot_freq_one.pdf", height=4, width=4)

cor.test(log(fcorrplot_df$count), fcorrplot_df$sum_syn_surprisal)

#---------------------------------------
# Convert surprisals to RTs 

fbound$word_pos <- fbound$word_pos + 1 # 0idx to 1idx
fmerged <- merge(fspr, fbound, by.x = c("Sentence", "WordPosition"), 
                 by.y=c("Sentence", "word_pos"), all.x=TRUE)
fmerged$logfreq <- log(fmerged$count)
fmerged$length <- nchar(fmerged$word)

print(unique(fmerged$word == fmerged$EachWord)) # Ensure correct alignment after merged

bound$word_clean <- str_replace_all(tolower(bound$word), "[.,!?:;]", "")
merged <- merge(bound, freqs, by.x = "word_clean", by.y="word", all.x=TRUE)
merged$word_pos <- merged$word_pos + 1 # 0idx to 1idx
merged <- merge(spr, merged, by.x = c("Sentence", "WordPosition"), 
                 by.y=c("Sentence", "word_pos"), all.x=TRUE)
merged$logfreq <- log(merged$count)
merged$length <- nchar(merged$word)

print(unique(merged$word == merged$EachWord)) #ensure correct alignment after merge

syn_surp_mean <- mean(c(fmerged$sum_syn_surprisal, merged$sum_syn_surprisal), na.rm=TRUE)
syn_surp_sd <- sd(c(fmerged$sum_syn_surprisal, merged$sum_syn_surprisal), na.rm=TRUE)
merged$syn_surprisal_s <- (merged$sum_syn_surprisal - syn_surp_mean)/syn_surp_sd
fmerged$syn_surprisal_s <- (fmerged$sum_syn_surprisal - syn_surp_mean)/syn_surp_sd

lex_surp_mean <- mean(c(fmerged$sum_lex_surprisal, merged$sum_lex_surprisal), na.rm=TRUE)
lex_surp_sd <- sd(c(fmerged$sum_lex_surprisal, merged$sum_lex_surprisal), na.rm=TRUE)
merged$lex_surprisal_s <- (merged$sum_lex_surprisal - lex_surp_mean)/lex_surp_sd
fmerged$lex_surprisal_s <- (fmerged$sum_lex_surprisal - lex_surp_mean)/lex_surp_sd

len_mean <- mean(c(fmerged$length, merged$length), na.rm=TRUE)
len_sd <- sd(c(fmerged$length, merged$length), na.rm=TRUE)
merged$length_s <- (merged$length - len_mean)/len_sd
fmerged$length_s <- (fmerged$length - len_mean)/len_sd

lfreq_mean <- mean(c(fmerged$logfreq, merged$logfreq), na.rm=TRUE)
lfreq_sd <- sd(c(fmerged$logfreq, merged$logfreq), na.rm=TRUE)
merged$logfreq_s <- (merged$logfreq - lfreq_mean)/lfreq_sd
fmerged$logfreq_s <- (fmerged$logfreq - lfreq_mean)/lfreq_sd

fmerged$item <- fmerged$item.x
merged$item <- merged$item.x
saveRDS(merged, "RDS/merged.rds")

fwith_lags <- fmerged %>% group_by_at(c("item", "MD5", "model")) %>%
                         mutate(lex_surprisal_p1_s = lag(lex_surprisal_s),
                                lex_surprisal_p2_s = lag(lex_surprisal_p1_s),
                                lex_surprisal_p3_s = lag(lex_surprisal_p2_s),
                                syn_surprisal_p1_s = lag(syn_surprisal_s),
                                syn_surprisal_p2_s = lag(syn_surprisal_p1_s),
                                syn_surprisal_p3_s = lag(syn_surprisal_p2_s),
                                logfreq_p1_s = lag(logfreq_s),
                                logfreq_p2_s = lag(logfreq_p1_s),
                                logfreq_p3_s = lag(logfreq_p2_s),
                                length_p1_s = lag(length_s),
                                length_p2_s = lag(length_p1_s),
                                length_p3_s = lag(length_p2_s))

fwith_lags$sent_length <- lapply(str_split(fwith_lags$Sentence, " "), length)

fdropped <- fwith_lags %>% subset(!is.na(lex_surprisal_s) & !is.na(lex_surprisal_p1_s) &
                              !is.na(lex_surprisal_p2_s) & !is.na(lex_surprisal_p3_s) &
                              !is.na(syn_surprisal_s) & !is.na(syn_surprisal_p1_s) & 
                              !is.na(syn_surprisal_p2_s) & !is.na(syn_surprisal_p3_s) &
                              !is.na(logfreq_s) & !is.na(logfreq_p1_s) & 
                              !is.na(logfreq_p2_s) & !is.na(logfreq_p3_s) &
                              fwith_lags$sent_length != fwith_lags$WordPosition
                              )
print(nrow(fwith_lags))
print(nrow(fdropped))

rm(fwith_lags)
rm(fmerged)
rm(spr)
rm(fspr)
gc()

saveRDS(fdropped, "RDS/fdropped.rds")

fdropped_minimal <- fdropped[,c("RT", "syn_surprisal_s", "syn_surprisal_p1_s", "syn_surprisal_p2_s",
                               "lex_surprisal_s", "lex_surprisal_p1_s", "lex_surprisal_p2_s",
                               "WordPosition", "logfreq_s", "logfreq_p1_s", "logfreq_p2_s",
                               "length_s", "length_p1_s", "length_p2_s", "MD5", "item", "model")]
saveRDS(fdropped_minimal, "RDS/fdropped_minimal.rds")

# -----------------------

for (model_num in c("m0", "m1", "m2", "m3")) {
  fdropped <- readRDS("RDS/fdropped_minimal.rds")
  fdropped <- fdropped %>% subset(model == model_num)
  
  filler_model_none <- lmer(RT ~ 
                              logfreq_s*length_s + logfreq_p1_s*length_p1_s + 
                              logfreq_p2_s*length_p2_s +
                              (1 | MD5) +
                              (1 | item),
                            data = fdropped,
                            control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=1e5)))
  saveRDS(filler_model_none, paste0("models/filler_model_none", model_num, ".rds"))
  
  rm(filler_model_none)
  gc()
  
  filler_model_syn <- lmer(RT ~ syn_surprisal_s + syn_surprisal_p1_s +
                             syn_surprisal_p2_s + scale(WordPosition) +
                             logfreq_s*length_s + logfreq_p1_s*length_p1_s + 
                             logfreq_p2_s*length_p2_s +
                             (1 + syn_surprisal_s + syn_surprisal_p1_s + 
                                syn_surprisal_p2_s || MD5) +
                             (1 | item),
                           data = fdropped,
                           control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=1e5)))
  
  saveRDS(filler_model_syn, paste0("models/filler_model_syn", model_num, ".rds"))
  rm(filler_model_syn)
  gc()
  
  filler_model_lex <- lmer(RT ~ lex_surprisal_s + lex_surprisal_p1_s + 
                             lex_surprisal_p2_s +
                             scale(WordPosition) +
                             logfreq_s*length_s + logfreq_p1_s*length_p1_s + 
                             logfreq_p2_s*length_p2_s + 
                             (1 + lex_surprisal_s + lex_surprisal_p1_s + 
                                lex_surprisal_p2_s  || MD5) +
                             (1 | item),
                           data = fdropped,
                           control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=1e5)))
  
  saveRDS(filler_model_lex, paste0("models/filler_model_lex", model_num,".rds"))
  rm(filler_model_lex)
  gc()
  
  
  filler_model_both <- lmer(RT ~ lex_surprisal_s + lex_surprisal_p1_s + 
                         lex_surprisal_p2_s  +
                         syn_surprisal_s + syn_surprisal_p1_s +
                         syn_surprisal_p2_s  + scale(WordPosition) +
                         logfreq_s*length_s + logfreq_p1_s*length_p1_s + 
                         logfreq_p2_s*length_p2_s + 
                         (1 + lex_surprisal_s + lex_surprisal_p1_s + 
                            lex_surprisal_p2_s  +
                              syn_surprisal_s + syn_surprisal_p1_s + 
                            syn_surprisal_p2_s || MD5) +
                         (1 | item),
                       data = fdropped,
                       control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=1e5)))
  
  saveRDS(filler_model_both, paste0("models/filler_model_both", model_num, ".rds"))
  rm(filler_model_both)
  rm(fdropped)
  gc()
}
#---------

merged <- readRDS("RDS/merged.rds")
with_lags <- merged %>% group_by_at(c("item", "MD5", "model")) %>%
  mutate(lex_surprisal_p1_s = lag(lex_surprisal_s),
         lex_surprisal_p2_s = lag(lex_surprisal_p1_s),
         lex_surprisal_p3_s = lag(lex_surprisal_p2_s),
         syn_surprisal_p1_s = lag(syn_surprisal_s),
         syn_surprisal_p2_s = lag(syn_surprisal_p1_s),
         syn_surprisal_p3_s = lag(syn_surprisal_p2_s),
         logfreq_p1_s = lag(logfreq_s),
         logfreq_p2_s = lag(logfreq_p1_s),
         logfreq_p3_s = lag(logfreq_p2_s),
         length_p1_s = lag(length_s),
         length_p2_s = lag(length_p1_s),
         length_p3_s = lag(length_p2_s))
rm(merged)

with_lags_pred <- subset(with_lags, is.na(model))
summary(with_lags_pred)
for (model_num in c("m0")) {
  with_lags_m <- subset(with_lags, model == model_num)
  filler_model_none <- readRDS(paste0("models/filler_model_none", model_num,".rds"))
  with_lags_m$predicted_rt_none <- predict(filler_model_none, newdata=with_lags_m, allow.new.levels=TRUE)
  rm(filler_model_none)
  
  
  filler_model_lex <- readRDS(paste0("models/filler_model_lex", model_num,".rds"))
  with_lags_m$predicted_rt_lex <- predict(filler_model_lex, newdata=with_lags_m, allow.new.levels=TRUE)
  rm(filler_model_lex)
  
  
  filler_model_syn <- readRDS(paste0("models/filler_model_syn", model_num,".rds"))
  with_lags_m$predicted_rt_syn <- predict(filler_model_syn, newdata=with_lags_m, allow.new.levels=TRUE)
  rm(filler_model_syn)
  
  
  filler_model_both <- readRDS(paste0("models/filler_model_both", model_num,".rds"))
  with_lags_m$predicted_rt_both <- predict(filler_model_both, newdata=with_lags_m, allow.new.levels=TRUE)
  rm(filler_model_both)
  
  with_lags_pred <- rbind(with_lags_pred, with_lags_m)

}
rm(with_lags)
gc()
with_lags <- with_lags_pred %>% subset(!is.na(predicted_rt_lex)) %>%
                         gather("rt_type", "rt", 
                                c("RT", "predicted_rt_none", "predicted_rt_lex", "predicted_rt_syn", 
                                  "predicted_rt_both"))

saveRDS(with_lags, "RDS/classic_gp_data.rds")

#----------
with_lags <- readRDS("RDS/classic_gp_data.rds")
with_lags <- with_lags %>% filter(ROI == 0) 
with_lags <- subset(with_lags, rt_type != "RT")

eff_model_nps <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "NPS")),
                     )

eff_model_npz <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "NPZ")),
)

eff_model_mvrr <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "MVRR")),
)


print("DISAMB---NPS")
nps.emm <- emmeans(eff_model_nps, ~ AMBIG | rt_type)
nps.con <- contrast(nps.emm, interaction="pairwise")
pairs(nps.con, by=NULL)

print("DISAMB---NPZ")
npz.emm <- emmeans(eff_model_npz, ~ AMBIG | rt_type)
npz.con <- contrast(npz.emm, interaction="pairwise")
pairs(npz.con, by=NULL)

print("DISAMB---MVRR")
mvrr.emm <- emmeans(eff_model_mvrr, ~ AMBIG | rt_type)
mvrr.con <- contrast(mvrr.emm, interaction="pairwise")
pairs(mvrr.con, by=NULL)

#----------
with_lags <- readRDS("RDS/classic_gp_data.rds")
with_lags <- with_lags %>% filter(ROI == 1) 
with_lags <- subset(with_lags, rt_type != "RT")

eff_model_nps <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "NPS")),
)


eff_model_npz <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "NPZ")),
)


eff_model_mvrr <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                       data=subset(with_lags, (with_lags$CONSTRUCTION == "MVRR")),
)

print("SP1---NPS")
nps.emm <- emmeans(eff_model_nps, ~ AMBIG | rt_type)
nps.con <- contrast(nps.emm, interaction="pairwise")
pairs(nps.con, by=NULL)

print("SP1---NPZ")
npz.emm <- emmeans(eff_model_npz, ~ AMBIG | rt_type)
npz.con <- contrast(npz.emm, interaction="pairwise")
pairs(npz.con, by=NULL)

print("SP1---MVRR")
mvrr.emm <- emmeans(eff_model_mvrr, ~ AMBIG | rt_type)
mvrr.con <- contrast(mvrr.emm, interaction="pairwise")
pairs(mvrr.con, by=NULL)

#----------
with_lags <- readRDS("RDS/classic_gp_data.rds")
with_lags <- with_lags %>% filter(ROI == 2) 
with_lags <- subset(with_lags, rt_type != "RT")

eff_model_nps <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "NPS")),
)

eff_model_npz <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                      data=subset(with_lags, (with_lags$CONSTRUCTION == "NPZ")),
)

eff_model_mvrr <- lmer(rt ~ AMBIG * rt_type + (1 | item) + (1 | MD5),
                       data=subset(with_lags, (with_lags$CONSTRUCTION == "MVRR")),
)

print("SP2---NPS")
nps.emm <- emmeans(eff_model_nps, ~ AMBIG | rt_type)
nps.con <- contrast(nps.emm, interaction="pairwise")
pairs(nps.con, by=NULL)

print("SP2---NPS")
npz.emm <- emmeans(eff_model_npz, ~ AMBIG | rt_type)
npz.con <- contrast(npz.emm, interaction="pairwise")
pairs(npz.con, by=NULL)

print("SP2---NPS")
mvrr.emm <- emmeans(eff_model_mvrr, ~ AMBIG | rt_type)
mvrr.con <- contrast(mvrr.emm, interaction="pairwise")
pairs(mvrr.con, by=NULL)

#----------

with_lags <- readRDS("RDS/classic_gp_data.rds")
with_lags <- with_lags %>% filter(ROI >= -1) %>% filter(ROI < 3)

ggplot(data=with_lags, aes(x=ROI, y=rt, color=rt_type, group=interaction(rt_type, AMBIG), linetype=AMBIG)) +
  stat_summary(geom="point") + stat_summary(geom="errorbar", width=0.3) + 
  stat_summary(geom="line") + facet_grid(~CONSTRUCTION)
ggsave("plots/rts_predicted.pdf", height=6, width=8)

effs <- with_lags %>% group_by(ROI, CONSTRUCTION, item, AMBIG, rt_type, model) %>%
  summarize(rt = mean(rt)) %>%
  pivot_wider(id_cols = c("ROI", "CONSTRUCTION", "item", "rt_type", "model"),
              names_from = c("AMBIG"), values_from=c("rt")) %>%
  mutate(rt_diff = Amb-Unamb)

levels = c("predicted_rt_both", "predicted_rt_syn", "predicted_rt_lex", "predicted_rt_none", "RT")

ggplot(data=effs, aes(x=ROI, y=rt_diff, fill=factor(rt_type, levels=levels))) +
  stat_summary(geom="bar", position=position_dodge()) + 
  stat_summary(geom="errorbar", fun.data=mean_cl_boot, 
               position=position_dodge()) + facet_grid(~CONSTRUCTION) +
  theme_classic() + theme(axis.text.x=element_text(size=7,angle=30, hjust=1), 
                          legend.position="top", legend.text=element_text(size=7),
                          strip.text.x=element_text(size=7)) +
  scale_x_continuous(breaks=c(-1, 0, 1, 2), 
                     labels=c("Pre-disamb", "Disamb", "Spillover 1", "Spillover 2")) +
  scale_fill_brewer(labels=c("predicted_rt_both"="Both Surprisals", "predicted_rt_lex"="Lexical Surprisal Only", 
                               "predicted_rt_syn"="Syntactic Surprisal Only", "predicted_rt_none"="No Surprisals", 
                               "RT"="Human RTs"),
                       palette="Dark2") +
  labs(x="", y="Garden Path Effect (ms)", fill="")
ggsave("plots/effs_predicted.pdf", height=2, width=6)

ggplot(data=subset(effs, rt_type != "RT"), aes(x=ROI, y=rt_diff, fill=factor(rt_type, levels=levels))) +
  stat_summary(geom="bar", position=position_dodge()) + 
  stat_summary(geom="errorbar", fun.data=mean_cl_boot, 
               position=position_dodge()) + facet_grid(model~CONSTRUCTION) +
  theme_classic() + theme(axis.text.x=element_text(size=7,angle=30, hjust=1), 
                          legend.position="top", legend.text=element_text(size=7),
                          strip.text.x=element_text(size=7)) +
  scale_x_continuous(breaks=c(-1, 0, 1, 2), 
                     labels=c("Pre-disamb", "Disamb", "Spillover 1", "Spillover 2")) +
  scale_fill_brewer(labels=c("predicted_rt_both"="Both Surprisals", "predicted_rt_lex"="Lexical Surprisal Only", 
                               "predicted_rt_syn"="Syntactic Surprisal Only", "predicted_rt_none"="No Surprisals", 
                             "RT"="Human RTs"),
                    palette="Dark2") +
  labs(x="", y="Garden Path Effect (ms)", fill="")
ggsave("plots/effs_predicted_models.pdf", height=2.5, width=6)

#-------
