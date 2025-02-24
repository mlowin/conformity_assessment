-- MySQL dump 10.13  Distrib 8.0.32, for Linux (x86_64)
--
-- Host: localhost    Database: dss_jailbreak
-- ------------------------------------------------------
-- Server version	8.0.32-0ubuntu0.20.04.2

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `answer`
--

DROP TABLE IF EXISTS `answer`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `answer` (
  `id` varchar(100) NOT NULL,
  `question` varchar(100) NOT NULL,
  `description` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`id`,`question`),
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `answer`
--

LOCK TABLES `answer` WRITE;
/*!40000 ALTER TABLE `answer` DISABLE KEYS */;
INSERT INTO `answer` VALUES ('no_better','specific_task','No, but there are no better datasets available'),('no_various','specific_task','No, the dataset can be used for various tasks'),('yes_different_purpose','specific_task','Yes, but the model builds on a different purpose'),('yes_purpose','specific_task','Yes and the model builds on this purpose');
/*!40000 ALTER TABLE `answer` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `question`
--

DROP TABLE IF EXISTS `question`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `question` (
  `id` varchar(100) NOT NULL,
  `category` varchar(100) NOT NULL,
  `description` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `question`
--

LOCK TABLES `question` WRITE;
/*!40000 ALTER TABLE `question` DISABLE KEYS */;
INSERT INTO `question` VALUES ('specific_task','Dataset','Was a specific task in mind when the dataset was created?');
/*!40000 ALTER TABLE `question` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `usecase`
--

DROP TABLE IF EXISTS `usecase`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `usecase` (
  `id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(100) NOT NULL,
  `weights` varchar(10000) DEFAULT NULL,
  `description` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `title` (`title`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `usecase`
--

LOCK TABLES `usecase` WRITE;
/*!40000 ALTER TABLE `usecase` DISABLE KEYS */;
INSERT INTO `usecase` VALUES (1,'Test Use Case','{\"fairness\": {\"title\": \"Fairness\", \"children\": {\"stat_par_mean\": {\"title\": \"Statistical Parity Difference\", \"rules\": {\"active\": true, \"green_min\": \"0\", \"green_max\": \"0.2\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.4\"}}, \"eq_odds_mean\": {\"title\": \"Average Absolute Odds Difference\", \"rules\": {\"active\": true, \"green_min\": \"0\", \"green_max\": \"0.2\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.4\"}}}, \"description\": \"Does not exist\"}, \"explainability\": {\"title\": \"Explainability\", \"children\": {\"mean_stability\": {\"title\": \"Mean Stability\", \"rules\": {\"active\": true, \"green_min\": \"0.8\", \"green_max\": \"1\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.8\"}}}, \"description\": \"What?\"}, \"performance\": {\"title\": \"Performance\", \"children\": {\"accuracy_mean\": {\"title\": \"Accuracy\", \"rules\": {\"active\": true, \"green_min\": \"0.8\", \"green_max\": \"1\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.8\"}}, \"balanced_acc_mean\": {\"title\": \"Balanced Accuracy\", \"rules\": {\"active\": true, \"green_min\": \"0.8\", \"green_max\": \"1\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.8\"}}, \"precision_mean\": {\"title\": \"Precision\", \"rules\": {\"active\": true, \"green_min\": \"0.8\", \"green_max\": \"1\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.8\"}}, \"recall_mean\": {\"title\": \"Recall\", \"rules\": {\"active\": true, \"green_min\": \"0.8\", \"green_max\": \"1\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.8\"}}, \"f1_mean\": {\"title\": \"F1-Score\", \"rules\": {\"active\": true, \"green_min\": \"0.8\", \"green_max\": \"1\", \"yellow_min\": \"0.2\", \"yellow_max\": \"0.8\"}}}, \"description\": \"BWL High Performer\"}, \"questions\": {\"specific_task\": {\"no_better\": \"red\", \"no_various\": \"yellow\", \"yes_different_purpose\": \"yellow\", \"yes_purpose\": \"green\"}}}','Use this use case for demo evaluation, please.');
/*!40000 ALTER TABLE `usecase` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-04-20 13:55:36
