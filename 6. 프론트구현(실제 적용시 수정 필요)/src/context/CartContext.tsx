"use client";

import { createContext, useContext, useState, ReactNode } from "react";

// 장바구니 상품 타입
export interface CartProduct {
  code: string;
  name: string;
  weight_pct: number;
  productRegion?: string;
  productTheme?: string;
  productType?: string;
  isTDF?: boolean;
}

// 장바구니 포트폴리오 타입
export interface CartPortfolio {
  portfolioId: number;
  conditions: {
    region: string;
    theme: string;
    targetReturn: number;
    retireYear: number;
  };
  metrics: {
    expectedReturn: number;
    var95: number;
  };
  allocation: {
    riskAssetWeight: number;
    safeAssetWeight: number;
    tdfWeight: number;
  };
  products: CartProduct[];
  totalProducts: number;  // 전체 상품 수
  addedAt: Date;
}

interface CartContextType {
  cartItems: CartPortfolio[];
  addToCart: (portfolio: Omit<CartPortfolio, "addedAt">) => void;
  removeFromCart: (portfolioId: number) => void;
  clearCart: () => void;
  isInCart: (portfolioId: number) => boolean;
}

const CartContext = createContext<CartContextType | undefined>(undefined);

export function CartProvider({ children }: { children: ReactNode }) {
  const [cartItems, setCartItems] = useState<CartPortfolio[]>([]);

  const addToCart = (portfolio: Omit<CartPortfolio, "addedAt">) => {
    // 이미 장바구니에 있으면 추가하지 않음
    if (cartItems.some((item) => item.portfolioId === portfolio.portfolioId)) {
      return;
    }
    setCartItems((prev) => [...prev, { ...portfolio, addedAt: new Date() }]);
  };

  const removeFromCart = (portfolioId: number) => {
    setCartItems((prev) => prev.filter((item) => item.portfolioId !== portfolioId));
  };

  const clearCart = () => {
    setCartItems([]);
  };

  const isInCart = (portfolioId: number) => {
    return cartItems.some((item) => item.portfolioId === portfolioId);
  };

  return (
    <CartContext.Provider value={{ cartItems, addToCart, removeFromCart, clearCart, isInCart }}>
      {children}
    </CartContext.Provider>
  );
}

export function useCart() {
  const context = useContext(CartContext);
  if (context === undefined) {
    throw new Error("useCart must be used within a CartProvider");
  }
  return context;
}
