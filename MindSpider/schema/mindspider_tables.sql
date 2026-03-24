-- MindSpider AI crawler project - Database schema
-- Extends MediaCrawler schema with tables required by the BroadTopicExtraction module

-- ===============================
-- BroadTopicExtraction module schema
-- ===============================

-- ----------------------------
-- Table structure for daily_news
-- Daily news table: stores trending news collected by get_today_news.py
-- ----------------------------
DROP TABLE IF EXISTS `daily_news`;
CREATE TABLE `daily_news` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT 'Auto-increment ID',
    `news_id` varchar(128) NOT NULL COMMENT 'Unique news ID',
    `source_platform` varchar(32) NOT NULL COMMENT 'Source platform (weibo|zhihu|bilibili|toutiao|douyin etc.)',
    `title` varchar(500) NOT NULL COMMENT 'News title',
    `url` varchar(512) DEFAULT NULL COMMENT 'News URL',
    `description` text COMMENT 'News description or summary',
    `extra_info` text COMMENT 'Extra information (stored as JSON)',
    `crawl_date` date NOT NULL COMMENT 'Crawl date',
    `rank_position` int DEFAULT NULL COMMENT 'Ranking position in hot list',
    `add_ts` bigint NOT NULL COMMENT 'Record creation timestamp',
    `last_modify_ts` bigint NOT NULL COMMENT 'Record last-modified timestamp',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_daily_news_unique` (`news_id`, `source_platform`, `crawl_date`),
    KEY `idx_daily_news_date` (`crawl_date`),
    KEY `idx_daily_news_platform` (`source_platform`),
    KEY `idx_daily_news_rank` (`rank_position`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Daily trending news table';

-- ----------------------------
-- Table structure for daily_topics
-- Daily topics table: stores topics extracted by TopicGPT
-- ----------------------------
DROP TABLE IF EXISTS `daily_topics`;
CREATE TABLE `daily_topics` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT 'Auto-increment ID',
    `topic_id` varchar(64) NOT NULL COMMENT 'Unique topic ID',
    `topic_name` varchar(255) NOT NULL COMMENT 'Topic name',
    `topic_description` text COMMENT 'Topic description',
    `keywords` text COMMENT 'Topic keywords (stored as JSON)',
    `extract_date` date NOT NULL COMMENT 'Topic extraction date',
    `relevance_score` float DEFAULT NULL COMMENT 'Topic relevance score',
    `news_count` int DEFAULT 0 COMMENT 'Number of related news items',
    `processing_status` varchar(16) DEFAULT 'pending' COMMENT 'Processing status (pending|processing|completed|failed)',
    `add_ts` bigint NOT NULL COMMENT 'Record creation timestamp',
    `last_modify_ts` bigint NOT NULL COMMENT 'Record last-modified timestamp',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_daily_topics_unique` (`topic_id`, `extract_date`),
    KEY `idx_daily_topics_date` (`extract_date`),
    KEY `idx_daily_topics_status` (`processing_status`),
    KEY `idx_daily_topics_score` (`relevance_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Daily extracted topics table';

-- ----------------------------
-- Table structure for topic_news_relation
-- Topic-news relation table: records topic-to-news relationships
-- ----------------------------
DROP TABLE IF EXISTS `topic_news_relation`;
CREATE TABLE `topic_news_relation` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT 'Auto-increment ID',
    `topic_id` varchar(64) NOT NULL COMMENT 'Topic ID',
    `news_id` varchar(128) NOT NULL COMMENT 'News ID',
    `relation_score` float DEFAULT NULL COMMENT 'Relation score',
    `extract_date` date NOT NULL COMMENT 'Relation extraction date',
    `add_ts` bigint NOT NULL COMMENT 'Record creation timestamp',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_topic_news_unique` (`topic_id`, `news_id`, `extract_date`),
    KEY `idx_topic_news_topic` (`topic_id`),
    KEY `idx_topic_news_news` (`news_id`),
    KEY `idx_topic_news_date` (`extract_date`),
    FOREIGN KEY (`topic_id`) REFERENCES `daily_topics`(`topic_id`) ON DELETE CASCADE,
    FOREIGN KEY (`news_id`) REFERENCES `daily_news`(`news_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Topic-news relation table';

-- ----------------------------
-- Table structure for crawling_tasks
-- Crawling tasks table: records platform crawling tasks based on topics
-- ----------------------------
DROP TABLE IF EXISTS `crawling_tasks`;
CREATE TABLE `crawling_tasks` (
    `id` int NOT NULL AUTO_INCREMENT COMMENT 'Auto-increment ID',
    `task_id` varchar(64) NOT NULL COMMENT 'Unique task ID',
    `topic_id` varchar(64) NOT NULL COMMENT 'Related topic ID',
    `platform` varchar(32) NOT NULL COMMENT 'Target platform (xhs|dy|ks|bili|wb|tieba|zhihu)',
    `search_keywords` text NOT NULL COMMENT 'Search keywords (stored as JSON)',
    `task_status` varchar(16) DEFAULT 'pending' COMMENT 'Task status (pending|running|completed|failed|paused)',
    `start_time` bigint DEFAULT NULL COMMENT 'Task start timestamp',
    `end_time` bigint DEFAULT NULL COMMENT 'Task end timestamp',
    `total_crawled` int DEFAULT 0 COMMENT 'Total crawled content count',
    `success_count` int DEFAULT 0 COMMENT 'Successful crawl count',
    `error_count` int DEFAULT 0 COMMENT 'Error count',
    `error_message` text COMMENT 'Error message',
    `config_params` text COMMENT 'Crawl configuration parameters (JSON)',
    `scheduled_date` date NOT NULL COMMENT 'Scheduled execution date',
    `add_ts` bigint NOT NULL COMMENT 'Record creation timestamp',
    `last_modify_ts` bigint NOT NULL COMMENT 'Record last-modified timestamp',
    PRIMARY KEY (`id`),
    UNIQUE KEY `idx_crawling_tasks_unique` (`task_id`),
    KEY `idx_crawling_tasks_topic` (`topic_id`),
    KEY `idx_crawling_tasks_platform` (`platform`),
    KEY `idx_crawling_tasks_status` (`task_status`),
    KEY `idx_crawling_tasks_date` (`scheduled_date`),
    FOREIGN KEY (`topic_id`) REFERENCES `daily_topics`(`topic_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Crawling tasks table';

-- ===============================
-- MediaCrawler schema extension fields
-- ===============================

-- Add topic relation fields to existing MediaCrawler tables for MindSpider features
-- Note: these fields are optional and do not affect original MediaCrawler functionality

-- Add topic relation fields to Xiaohongshu note table
ALTER TABLE `xhs_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- Add topic relation fields to Douyin video table
ALTER TABLE `douyin_aweme` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- Add topic relation fields to Kuaishou video table
ALTER TABLE `kuaishou_video` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- Add topic relation fields to Bilibili video table
ALTER TABLE `bilibili_video` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- Add topic relation fields to Weibo post table
ALTER TABLE `weibo_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- Add topic relation fields to Tieba post table
ALTER TABLE `tieba_note` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- Add topic relation fields to Zhihu content table
ALTER TABLE `zhihu_content` 
ADD COLUMN `topic_id` varchar(64) DEFAULT NULL COMMENT 'Related topic ID',
ADD COLUMN `crawling_task_id` varchar(64) DEFAULT NULL COMMENT 'Related crawling task ID';

-- ===============================
-- Create views for data analysis
-- ===============================

-- Topic crawling statistics view
CREATE OR REPLACE VIEW `v_topic_crawling_stats` AS
SELECT 
    dt.topic_id,
    dt.topic_name,
    dt.extract_date,
    dt.processing_status,
    COUNT(DISTINCT ct.task_id) as total_tasks,
    SUM(CASE WHEN ct.task_status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
    SUM(CASE WHEN ct.task_status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
    SUM(ct.total_crawled) as total_content_crawled,
    SUM(ct.success_count) as total_success_count,
    SUM(ct.error_count) as total_error_count
FROM daily_topics dt
LEFT JOIN crawling_tasks ct ON dt.topic_id = ct.topic_id
GROUP BY dt.topic_id, dt.topic_name, dt.extract_date, dt.processing_status;

-- Daily data summary view
CREATE OR REPLACE VIEW `v_daily_summary` AS
SELECT 
    crawl_date,
    COUNT(DISTINCT news_id) as total_news,
    COUNT(DISTINCT source_platform) as platforms_covered,
    (SELECT COUNT(*) FROM daily_topics WHERE extract_date = dn.crawl_date) as topics_extracted,
    (SELECT COUNT(*) FROM crawling_tasks WHERE scheduled_date = dn.crawl_date) as tasks_created
FROM daily_news dn
GROUP BY crawl_date
ORDER BY crawl_date DESC;

-- ===============================
-- Initial index optimization
-- ===============================

-- Add composite indexes for relation-query optimization
CREATE INDEX `idx_topic_date_status` ON `daily_topics` (`extract_date`, `processing_status`);
CREATE INDEX `idx_task_topic_platform` ON `crawling_tasks` (`topic_id`, `platform`, `task_status`);
CREATE INDEX `idx_news_date_platform` ON `daily_news` (`crawl_date`, `source_platform`);

-- ===============================
-- Database configuration optimization suggestions
-- ===============================

-- Set an appropriate charset and collation
-- ALTER DATABASE mindspider CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Suggested data retention policy (optional)
-- You can create an event scheduler to clean historical data as needed
-- Example: delete news data older than 90 days while retaining topic and crawl result data
