const request = async (
    url: string,
    method: "GET" | "POST" | "PUT" | "DELETE",
    options?: Record<string, any>,
  ) => {
    const baseUrl = `http://localhost:3000${url}`;
  
    const payload =
      method === "GET"
        ? {
            method,
          }
        : {
            method,
            headers: {
              "Content-type": "application/json",
            },
            body: JSON.stringify(options),
          };
    const res = await fetch(baseUrl, { ...payload, cache: "no-store" });
    if (!res.ok) {
      return { errorCode: res.status, errorMessage: res.statusText };
    }
    return await res.json();
  };

export const get = (url: string, options?: Record<string, string>) =>
  request(url, "GET", options);

export const post = (url: string, options?: Record<string, any>) =>
  request(url, "POST", options);
