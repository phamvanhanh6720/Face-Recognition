create table datasets(
	id int primary key,
	name text not null,
	cropped_images bytea,
	embeddings bytea,
	latest_update date default CURRENT_DATE);
	
	
select * from datasets;



