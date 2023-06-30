CREATE TABLE `question_answer` (
  `id` int NOT NULL AUTO_INCREMENT,
  `question` varchar(512) NOT NULL,
  `answer` varchar(512) NOT NULL,
  `answer_html` longtext NOT NULL,
  PRIMARY KEY (`id`)
)