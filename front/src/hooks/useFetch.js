import { useEffect, useState } from "react";

export default function useFetch(URL) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [URL]);

  // functions
  async function fetchData() {
    try {
      setLoading(true);
      const res = await fetch(URL);
      if (res.ok) {
        const parsedData = await res.json();
        setData(parsedData);
      } else {
        throw new Error("Unable to fetch the data requested.");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return { data, error, loading };
}
