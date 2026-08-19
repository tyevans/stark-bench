"""Use cases: what this benchmark *does*, with nothing about how.

A module belongs here when it orchestrates ports to produce a domain value
and names no technology. The test for membership is blunt: if it mentions
Postgres, Neo4j, a DSN, a file path or an HTTP endpoint, it belongs in
`adapters` or `composition` instead.
"""
