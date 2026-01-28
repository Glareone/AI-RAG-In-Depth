# RAPTOR. Hierarchical Indexing.
## Description. Goal
This process provides you with document information of varying granularity. 
You obtain a high-level summary, then mid-level summaries, and ultimately highly specific chunks. 

All the retrieved chunks provide a comprehensive context for the generation step.

## Options
Hierarchical chunking is a simplified version of a GraphRAG.

## Details
Hierarchical indexing involves organizing content at multiple levels.  
The text resource is broken down from a document into sections, then paragraphs, and finally chunks.  
This strategy respects a document's structure and does not randomly create chunks.  

## How to implement
<img width="1160" height="504" alt="image" src="https://github.com/user-attachments/assets/a67f3414-4bd9-41a8-8e48-e2145d5f51ce" />

During the inference step (see Figure 3-10), you walk the tree to the final chunks:  
1) First you find the nearest embedding of first-level node.  
2) Then, you match against all second-level nodes that correspond to the first level node you chose, and so on.  
3) Each chunk and summary node will need its own embedding, so the process is recursive and isn’t a single-step embedding process.  

<img width="1163" height="725" alt="image" src="https://github.com/user-attachments/assets/706b9469-3fba-4be8-9503-395150d3af06" />

